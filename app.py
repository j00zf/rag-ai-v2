import os
import uuid
import logging

from flask import Flask, render_template, request, jsonify, redirect, session
from dotenv import load_dotenv

import mysql.connector

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from groq import Groq


# -------------------------
# Load ENV
# -------------------------

load_dotenv()

# -------------------------
# Create Required Folders
# -------------------------

os.makedirs("pdfs", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -------------------------
# Logging Setup
# -------------------------

# RAG / AI errors
rag_logger = logging.getLogger("rag_logger")
rag_logger.setLevel(logging.ERROR)

rag_handler = logging.FileHandler("logs/rag.error.log")
rag_handler.setFormatter(logging.Formatter(
"%(asctime)s - %(levelname)s - %(message)s"
))

rag_logger.addHandler(rag_handler)

# Backend / MySQL errors
app_logger = logging.getLogger("app_logger")
app_logger.setLevel(logging.ERROR)

app_handler = logging.FileHandler("logs/app-mysql.error.log")
app_handler.setFormatter(logging.Formatter(
"%(asctime)s - %(levelname)s - %(message)s"
))

app_logger.addHandler(app_handler)

# -------------------------
# Flask App
# -------------------------

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

session_id = str(uuid.uuid4())

# -------------------------
# MySQL Connection
# -------------------------

try:

    db = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        port=3307
    )

    cursor = db.cursor()

except Exception as e:

    app_logger.error(f"MySQL connection error: {str(e)}")

# -------------------------
# Groq Client
# -------------------------

try:

    groq_client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

except Exception as e:

    rag_logger.error(f"Groq client initialization error: {str(e)}")

# -------------------------
# Embeddings + Vector DB
# -------------------------

try:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

except Exception as e:

    rag_logger.error(f"Vector DB initialization error: {str(e)}")

# -------------------------
# PDF Processing
# -------------------------

def process_pdf(path, document_id):

    try:

        loader = PyPDFLoader(path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(documents)

        for doc in docs:

            text = doc.page_content

            cursor.execute(
                "INSERT INTO document_chunks (document_id,content) VALUES (%s,%s)",
                (document_id, text)
            )

        db.commit()

        vector_db.add_documents(docs)

        vector_db.persist()

    except Exception as e:

        rag_logger.error(f"PDF processing error: {str(e)}")

# -------------------------
# RAG Query
# -------------------------

def ask_bot(question):

    try:

        docs = retriever.get_relevant_documents(question)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an AI assistant for a college website.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content

    except Exception as e:

        rag_logger.error(f"RAG query error: {str(e)}")

        return "The AI assistant is currently unavailable."

# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():

    return render_template("chat.html")

# -------------------------
# Chat API
# -------------------------

@app.route("/chat", methods=["POST"])
def chat():

    try:

        message = request.json["message"]

        response = ask_bot(message)

        cursor.execute("""
        INSERT INTO chat_logs
        (session_id,user_message,bot_response)
        VALUES (%s,%s,%s)
        """, (session_id, message, response))

        db.commit()

        return jsonify({"reply": response})

    except Exception as e:

        app_logger.error(f"Chat route error: {str(e)}")

        return jsonify({"reply": "Server error occurred."})

# -------------------------
# ADMIN LOGIN
# -------------------------

@app.route("/admin")
def admin():

    return render_template("admin_login.html")

@app.route("/admin/login", methods=["POST"])
def admin_login():

    try:

        username = request.form["username"]
        password = request.form["password"]

        cursor.execute(
            "SELECT * FROM admins WHERE username=%s AND password=%s",
            (username, password)
        )

        admin = cursor.fetchone()

        if admin:

            session["admin"] = username

            return redirect("/admin/dashboard")

        return "Invalid login"

    except Exception as e:

        app_logger.error(f"Admin login error: {str(e)}")

        return "Login error"

# -------------------------
# ADMIN DASHBOARD
# -------------------------

@app.route("/admin/dashboard")
def admin_dashboard():

    if "admin" not in session:

        return redirect("/admin")

    return render_template("admin_dashboard.html")

# -------------------------
# Upload PDF
# -------------------------

@app.route("/admin/upload", methods=["POST"])
def upload_pdf():

    try:

        if "admin" not in session:

            return redirect("/admin")

        pdf = request.files["pdf"]

        path = "pdfs/" + pdf.filename

        pdf.save(path)

        cursor.execute(
            "INSERT INTO documents (filename) VALUES (%s)",
            (pdf.filename,)
        )

        db.commit()

        document_id = cursor.lastrowid

        process_pdf(path, document_id)

        return "PDF uploaded and processed successfully."

    except Exception as e:

        app_logger.error(f"PDF upload error: {str(e)}")

        return "Error uploading PDF."

# -------------------------
# View Chats
# -------------------------

@app.route("/admin/chats")
def view_chats():

    try:

        if "admin" not in session:

            return redirect("/admin")

        cursor.execute(
            "SELECT * FROM chat_logs ORDER BY created_at DESC"
        )

        chats = cursor.fetchall()

        return render_template("admin_chats.html", chats=chats)

    except Exception as e:

        app_logger.error(f"View chats error: {str(e)}")

        return "Error loading chats."

# -------------------------
# Run Server
# -------------------------

if __name__ == "__main__":

    app.run(debug=True)