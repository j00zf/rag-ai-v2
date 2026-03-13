import os
import uuid
import logging
import requests
from werkzeug.security import generate_password_hash, check_password_hash
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

def setup_logger(name, log_file):

    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)

    handler = logging.FileHandler(log_file)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


app_logger = setup_logger("app_logger", "logs/app.error.log")
rag_logger = setup_logger("rag_logger", "logs/rag.error.log")


# -------------------------
# Flask App
# -------------------------

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

session_id = str(uuid.uuid4())


# -------------------------
# Database Connection
# -------------------------

def get_db_connection():

    try:

        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB"),
            port=3307
        )

        return connection

    except Exception as e:

        app_logger.error(f"MySQL connection error: {str(e)}")

        return None


# -------------------------
# GROQ Client
# -------------------------

try:

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

except Exception as e:

    rag_logger.error(f"GROQ init error: {str(e)}")
    groq_client = None


# -------------------------
# Vector Database
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

    rag_logger.error(f"Vector DB init error: {str(e)}")
    vector_db = None
    retriever = None


# -------------------------
# Get User IP
# -------------------------

def get_user_ip():

    try:

        if request.headers.get("X-Forwarded-For"):
            return request.headers.get("X-Forwarded-For")

        return request.remote_addr

    except Exception as e:

        app_logger.error(f"IP fetch error: {str(e)}")
        return "unknown"


# -------------------------
# Get IP Location
# -------------------------

def get_ip_location(ip):

    try:

        response = requests.get(
            f"http://ip-api.com/json/{ip}",
            timeout=5
        ).json()

        return {
            "country": response.get("country"),
            "region": response.get("regionName"),
            "city": response.get("city"),
            "lat": response.get("lat"),
            "lon": response.get("lon")
        }

    except Exception as e:

        app_logger.error(f"IP lookup error: {str(e)}")
        return None


# -------------------------
# PDF Processing
# -------------------------

def process_pdf(path, document_id):

    if not vector_db:

        rag_logger.error("Vector DB not initialized")
        return

    try:

        loader = PyPDFLoader(path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(documents)

        db = get_db_connection()

        if db:

            cursor = db.cursor()

            for doc in docs:

                cursor.execute(
                    "INSERT INTO document_chunks (document_id,content) VALUES (%s,%s)",
                    (document_id, doc.page_content)
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

        if not retriever or not groq_client:

            return "AI system is currently unavailable."

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

        rag_logger.error(
            f"RAG error | Question: {question} | Error: {str(e)}"
        )

        return "AI assistant is temporarily unavailable."


# -------------------------
# Home
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

        message = request.json.get("message")

        if not message:

            return jsonify({"reply": "Empty message"})


        ip = get_user_ip()

        db = get_db_connection()

        if not db:

            return jsonify({"reply": "Database unavailable."})

        cursor = db.cursor()

        cursor.execute(
            "SELECT id FROM ip_addresses WHERE ip_address=%s",
            (ip,)
        )

        ip_row = cursor.fetchone()

        if ip_row:

            ip_id = ip_row[0]

        else:

            location = get_ip_location(ip)

            if location:

                cursor.execute("""
                INSERT INTO ip_addresses
                (ip_address,country,region,city,latitude,longitude)
                VALUES (%s,%s,%s,%s,%s,%s)
                """, (
                    ip,
                    location["country"],
                    location["region"],
                    location["city"],
                    location["lat"],
                    location["lon"]
                ))

                db.commit()

                ip_id = cursor.lastrowid

            else:

                ip_id = None


        response = ask_bot(message)

        cursor.execute("""
        INSERT INTO chat_logs
        (session_id,ip_id,user_message,bot_response)
        VALUES (%s,%s,%s,%s)
        """, (
            session_id,
            ip_id,
            message,
            response
        ))

        db.commit()

        return jsonify({"reply": response})

    except Exception as e:

        app_logger.error(f"/chat route error: {str(e)}")

        return jsonify({"reply": "Server error occurred."})


# ------------------------- 
# ADMIN REGISTER
# -------------------------

@app.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    if request.method == "GET":
        return render_template("admin_register.html")

    # POST handling
    try:
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        secret_key = request.form.get("secret_key", "").strip()   # optional extra protection

        if not username or not password:
            return render_template("admin_register.html", 
                                 error="Username and password are required")

        if password != confirm_password:
            return render_template("admin_register.html", 
                                 error="Passwords do not match")

        # Optional: protect registration with a secret code from .env
        if os.getenv("ADMIN_REGISTRATION_KEY") and secret_key != os.getenv("ADMIN_REGISTRATION_KEY"):
            return render_template("admin_register.html", 
                                 error="Invalid registration key")

        db = get_db_connection()
        if not db:
            return render_template("admin_register.html", 
                                 error="Database unavailable")

        cursor = db.cursor()

        # Check if username already exists
        cursor.execute("SELECT id FROM admins WHERE username = %s", (username,))
        if cursor.fetchone():
            return render_template("admin_register.html", 
                                 error="Username already taken")

        hashed_pw = generate_password_hash(password)

        cursor.execute("""
            INSERT INTO admins 
            (username, password, status, role, created_at)
            VALUES (%s, %s, 'inactive', 'admin', NOW())
        """, (username, hashed_pw))

        db.commit()

        return render_template("admin_register.html", 
                             success="Registration successful! Wait for superadmin approval.")

    except Exception as e:
        app_logger.error(f"Admin registration error: {str(e)}")
        return render_template("admin_register.html", 
                             error="Server error during registration")
    
# -------------------------
# ADMIN LOGIN
# -------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None

    if request.method == "POST":
        try:
            # Use .get() → safer, returns None instead of raising KeyError/400
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()

            if not username or not password:
                error = "Please enter both username and password"
            else:
                db = get_db_connection()
                if not db:
                    error = "Database is temporarily unavailable"
                else:
                    cursor = db.cursor()
                    cursor.execute(
                        "SELECT id, password, status FROM admins WHERE username = %s",
                        (username,)
                    )
                    admin = cursor.fetchone()

                    if not admin:
                        error = "Invalid username or password"
                    else:
                        admin_id, hashed_password, status = admin

                        if status != "active":
                            error = "This account has not been activated yet"
                        elif check_password_hash(hashed_password, password):
                            session["admin"] = admin_id
                            cursor.execute(
                                "UPDATE admins SET last_login = NOW() WHERE id = %s",
                                (admin_id,)
                            )
                            db.commit()
                            return redirect("/admin/dashboard")
                        else:
                            error = "Invalid username or password"

        except Exception as e:
            app_logger.error(f"/admin/login error: {str(e)}")
            error = "An unexpected error occurred. Please try again."

    # GET request OR failed POST → show form + any error
    return render_template("admin_login.html", error=error)

# -------------------------
# ADMIN DASHBOARD
# -------------------------
@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect("/admin/login")

    try:
        db = get_db_connection()
        if not db:
            return render_template("admin_dashboard.html", 
                                 error="Database connection failed")

        cursor = db.cursor(dictionary=True)

        # Basic stats
        stats = {}

        # Total uploaded documents
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        stats['total_documents'] = cursor.fetchone()['count']

        # Total chat messages (user + bot)
        cursor.execute("SELECT COUNT(*) as count FROM chat_logs")
        stats['total_messages'] = cursor.fetchone()['count']

        # Recent chats (last 10)
        cursor.execute("""
            SELECT 
                chat_logs.id,
                chat_logs.created_at,
                ip_addresses.ip_address,
                ip_addresses.city,
                ip_addresses.country,
                chat_logs.user_message,
                chat_logs.bot_response
            FROM chat_logs
            LEFT JOIN ip_addresses ON chat_logs.ip_id = ip_addresses.id
            ORDER BY chat_logs.created_at DESC
            LIMIT 10
        """)
        recent_chats = cursor.fetchall()

        # Most active IPs (top 5)
        cursor.execute("""
            SELECT 
                ip_addresses.ip_address,
                ip_addresses.city,
                ip_addresses.country,
                COUNT(*) as message_count
            FROM chat_logs
            JOIN ip_addresses ON chat_logs.ip_id = ip_addresses.id
            GROUP BY ip_addresses.id
            ORDER BY message_count DESC
            LIMIT 5
        """)
        top_ips = cursor.fetchall()

        return render_template("admin_dashboard.html",
                             stats=stats,
                             recent_chats=recent_chats,
                             top_ips=top_ips)

    except Exception as e:
        app_logger.error(f"Dashboard error: {str(e)}")
        return render_template("admin_dashboard.html", 
                             error="Error loading dashboard data")
# -------------------------
# Upload PDF
# -------------------------

@app.route("/admin/upload", methods=["POST"])
def upload_pdf():

    try:

        if "admin" not in session:

            return redirect("/")

        pdf = request.files["pdf"]

        path = "pdfs/" + pdf.filename

        pdf.save(path)

        db = get_db_connection()

        if not db:

            return "Database unavailable"

        cursor = db.cursor()

        cursor.execute(
            "INSERT INTO documents (filename) VALUES (%s)",
            (pdf.filename,)
        )

        db.commit()

        document_id = cursor.lastrowid

        process_pdf(path, document_id)

        return "PDF uploaded successfully."

    except Exception as e:

        app_logger.error(f"/admin/upload error: {str(e)}")

        return "Upload error"

# ────────────────────────────────────────────────
# MANAGE KNOWLEDGE BASE (list + basic actions)
# ────────────────────────────────────────────────
@app.route("/admin/knowledge-base")
def admin_knowledge_base():
    if "admin" not in session:
        return redirect("/admin/login")

    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, filename, uploaded_at FROM documents ORDER BY uploaded_at DESC")
        documents = cursor.fetchall()
        return render_template("admin_knowledge_base.html", documents=documents)
    except Exception as e:
        app_logger.error(f"Knowledge base error: {str(e)}")
        return render_template("admin_knowledge_base.html", error=str(e))


# ────────────────────────────────────────────────
# VIEW ALL IP ADDRESSES
# ────────────────────────────────────────────────
@app.route("/admin/ip-addresses")
def admin_ip_addresses():
    if "admin" not in session:
        return redirect("/admin/login")

    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                ip_address, country, region, city, latitude, longitude, 
                COUNT(chat_logs.id) as message_count,
                MAX(chat_logs.created_at) as last_seen
            FROM ip_addresses
            LEFT JOIN chat_logs ON ip_addresses.id = chat_logs.ip_id
            GROUP BY ip_addresses.id
            ORDER BY last_seen DESC
        """)
        ips = cursor.fetchall()
        return render_template("admin_ip_addresses.html", ips=ips)
    except Exception as e:
        app_logger.error(f"IP addresses error: {str(e)}")
        return render_template("admin_ip_addresses.html", error=str(e))


# ────────────────────────────────────────────────
# MANAGE ADMINS (list + activate/deactivate)
# ────────────────────────────────────────────────
@app.route("/admin/manage-admins")
def admin_manage_admins():
    if "admin" not in session:
        return redirect("/admin/login")

    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, username, status, role, created_at, last_login 
            FROM admins 
            ORDER BY created_at DESC
        """)
        admins = cursor.fetchall()
        return render_template("admin_manage_admins.html", admins=admins)
    except Exception as e:
        app_logger.error(f"Manage admins error: {str(e)}")
        return render_template("admin_manage_admins.html", error=str(e))


# Simple toggle status endpoint (POST)
@app.route("/admin/toggle-admin-status", methods=["POST"])
def toggle_admin_status():
    if "admin" not in session:
        return jsonify({"success": False, "message": "Not authorized"}), 403

    try:
        admin_id = request.form.get("admin_id")
        new_status = request.form.get("status")  # 'active' or 'inactive'

        if not admin_id or new_status not in ['active', 'inactive']:
            return jsonify({"success": False, "message": "Invalid parameters"}), 400

        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute(
            "UPDATE admins SET status = %s WHERE id = %s",
            (new_status, admin_id)
        )
        db.commit()
        return jsonify({"success": True})
    except Exception as e:
        app_logger.error(f"Toggle admin status error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


# ────────────────────────────────────────────────
# VIEW ERRORS / LOGS (simple file viewer - last N lines)
# ────────────────────────────────────────────────
@app.route("/admin/view-errors")
def admin_view_errors():
    if "admin" not in session:
        return redirect("/admin/login")

    try:
        log_files = {
            "App Errors": "logs/app.error.log",
            "RAG Errors": "logs/rag.error.log"
        }
        logs = {}

        for name, path in log_files.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-80:]   # last 80 lines
                    logs[name] = "".join(lines) or "(empty or no errors yet)"
            except FileNotFoundError:
                logs[name] = "(log file not found)"
            except Exception as e:
                logs[name] = f"Error reading log: {str(e)}"

        return render_template("admin_view_errors.html", logs=logs)
    except Exception as e:
        app_logger.error(f"View errors page error: {str(e)}")
        return "Error loading logs page", 500
# -------------------------
# View Chats
# -------------------------

@app.route("/admin/chats")
def view_chats():

    try:

        if "admin" not in session:

            return redirect("/")

        db = get_db_connection()

        if not db:

            return "Database unavailable"

        cursor = db.cursor()

        cursor.execute("""
        SELECT chat_logs.id, ip_addresses.ip_address,
        ip_addresses.country, ip_addresses.city,
        chat_logs.user_message, chat_logs.bot_response,
        chat_logs.created_at
        FROM chat_logs
        JOIN ip_addresses
        ON chat_logs.ip_id = ip_addresses.id
        ORDER BY chat_logs.created_at DESC
        """)

        chats = cursor.fetchall()

        return render_template(
            "admin_chats.html",
            chats=chats
        )

    except Exception as e:

        app_logger.error(f"/admin/chats error: {str(e)}")

        return "Error loading chats"


# -------------------------
# Global Error Handlers
# -------------------------

@app.errorhandler(404)
def not_found(e):

    app_logger.error(f"404 error: {request.url}")

    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):

    app_logger.error(f"500 error: {str(e)}")

    return render_template("500.html"), 500




# -------------------------
# Run Server
# -------------------------

if __name__ == "__main__":

    app.run(debug=True)
