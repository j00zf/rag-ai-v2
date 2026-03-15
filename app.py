import os
import uuid
import logging
import requests
import time
import traceback
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, redirect, session
from dotenv import load_dotenv
import mysql.connector

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from groq import Groq

# ────────────────────────────────────────────────
# Load environment variables
# ────────────────────────────────────────────────
load_dotenv()

# ────────────────────────────────────────────────
# Create required directories
# ────────────────────────────────────────────────
os.makedirs("pdfs", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────
def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

app_logger = setup_logger("app_logger", "logs/app.log")
rag_logger = setup_logger("rag_logger", "logs/rag.log")

# ────────────────────────────────────────────────
# Flask application
# ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or "fallback-secret-key-change-me-please"

session_id = str(uuid.uuid4())

# ────────────────────────────────────────────────
# Database connection
# ────────────────────────────────────────────────
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DB", "college_chat_bot"),
            port=int(os.getenv("MYSQL_PORT", 3306))
        )
        return conn
    except Exception as e:
        app_logger.error(f"MySQL connection error: {str(e)}")
        return None

# ────────────────────────────────────────────────
# Groq client
# ────────────────────────────────────────────────
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    app_logger.info("Groq client initialized successfully")
except Exception as e:
    rag_logger.error(f"GROQ init error: {str(e)}")
    groq_client = None

# ────────────────────────────────────────────────
# Lazy Vector DB initialization
# ────────────────────────────────────────────────
def get_vector_db():
    if hasattr(get_vector_db, "initialized") and get_vector_db.initialized:
        return get_vector_db.vector_db, get_vector_db.retriever

    try:
        app_logger.info("[VECTOR] Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        app_logger.info("[VECTOR] Connecting to Chroma...")
        vector_db = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings,
            collection_name="rag_documents"
        )

        count = vector_db._collection.count()
        app_logger.info(f"[VECTOR] Chroma ready — {count} documents in collection")

        retriever = vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 6, "score_threshold": 0.26}
        )

        get_vector_db.vector_db = vector_db
        get_vector_db.retriever = retriever
        get_vector_db.initialized = True

        return vector_db, retriever

    except Exception as e:
        msg = f"[VECTOR INIT FAILED] {str(e)}\n{traceback.format_exc()}"
        rag_logger.critical(msg)
        app_logger.critical(msg)
        return None, None

# ────────────────────────────────────────────────
# PDF processing – FIXED VERSION
# ────────────────────────────────────────────────
def process_pdf(path, document_id):
    app_logger.info(f"[PDF] Processing file: {path}")
    vector_db, _ = get_vector_db()
    
    if vector_db is None:
        rag_logger.error("Vector DB not initialized – cannot process PDF")
        return

    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        # Save to MySQL
        db = get_db_connection()
        if db:
            cursor = db.cursor()
            for idx, doc in enumerate(docs):
                cursor.execute(
                    "INSERT INTO document_chunks (document_id, content, chunk_index) VALUES (%s, %s, %s)",
                    (document_id, doc.page_content, idx)
                )
            db.commit()
            app_logger.info(f"[PDF] Saved {len(docs)} chunks to MySQL")

        # Prepare metadata
        metadata_list = [
            {
                "document_id": str(document_id),
                "chunk_index": i,
                "source": os.path.basename(path)
            }
            for i in range(len(docs))
        ]

        # Attach metadata to documents (correct way)
        for doc, meta in zip(docs, metadata_list):
            doc.metadata.update(meta)

        # Add to Chroma – correct call
        vector_db.add_documents(docs)
        vector_db.persist()

        new_count = vector_db._collection.count()
        app_logger.info(f"[PDF] Successfully added {len(docs)} chunks → total documents now: {new_count}")

    except Exception as e:
        rag_logger.error(f"[PDF] Processing failed: {str(e)}", exc_info=True)
        app_logger.error(f"[PDF] Processing failed: {str(e)}", exc_info=True)

# ────────────────────────────────────────────────
# RAG – stricter prompt
# ────────────────────────────────────────────────
def ask_bot(question):
    _, retriever = get_vector_db()
    if not retriever or not groq_client:
        return "The AI system is temporarily unavailable."

    try:
        docs = retriever.invoke(question)
        if not docs:
            return "I don't have information about that in the college documents."

        context = "\n\n".join([f"[Excerpt]:\n{doc.page_content}" for doc in docs])

        prompt = f"""You are a helpful college information assistant.
Answer **only** using the provided excerpts from official college documents.
Do NOT use your general knowledge. Do NOT make up information.
If nothing relevant is found, reply exactly:
"I don't have information about that in the available college documents."

Excerpts:
{context}

Question: {question}

Answer concisely, politely, and accurately.
Use bullet points or numbered lists when helpful."""

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=1200,
            top_p=0.92
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        rag_logger.error(f"RAG failed | q: {question[:80]}... | err: {str(e)}")
        return "Sorry, I'm having trouble accessing the knowledge base."

# ────────────────────────────────────────────────
# IP Helpers
# ────────────────────────────────────────────────
def get_user_ip():
    try:
        if request.headers.get("X-Forwarded-For"):
            return request.headers.get("X-Forwarded-For").split(",")[0].strip()
        return request.remote_addr
    except Exception as e:
        app_logger.error(f"IP fetch error: {str(e)}")
        return "unknown"

def get_ip_location(ip):
    api_key = os.getenv("IPGEOLOCATION_API_KEY")
    if not api_key:
        return None

    if ip in ["127.0.0.1", "::1", "unknown", "0.0.0.0"] or ip.startswith(("192.168.", "10.")):
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = f"https://api.ipgeolocation.io/ipgeo?apiKey={api_key}&ip={ip}"
            response = requests.get(url, timeout=8)
            if response.status_code != 200:
                return None
            data = response.json()
            if "message" in data and data["message"]:
                return None
            return {
                "country": data.get("country_name", ""),
                "region": data.get("state_prov", ""),
                "city": data.get("city", ""),
                "lat": str(data.get("latitude", "")) if data.get("latitude") is not None else None,
                "lon": str(data.get("longitude", "")) if data.get("longitude") is not None else None,
            }
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None

# ────────────────────────────────────────────────
# Public routes
# ────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        message = request.json.get("message")
        if not message:
            return jsonify({"reply": "Please enter a message"}), 400

        vector_db, retriever = get_vector_db()
        if not retriever:
            return jsonify({"reply": "Knowledge base not loaded"}), 503

        count = vector_db._collection.count() if vector_db else 0
        if count == 0:
            return jsonify({"reply": "No documents uploaded yet."})

        ip = get_user_ip()
        app_logger.info(f"[CHAT] IP: {ip} | docs: {count}")

        db = get_db_connection()
        if not db:
            return jsonify({"reply": "Database unavailable"}), 503

        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT id FROM ip_addresses WHERE ip_address = %s", (ip,))
        ip_row = cursor.fetchone()

        if ip_row:
            ip_id = ip_row['id']
        else:
            location = get_ip_location(ip)
            cursor.execute("""
                INSERT INTO ip_addresses
                (ip_address, country, region, city, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                ip,
                location.get("country") if location else None,
                location.get("region") if location else None,
                location.get("city") if location else None,
                location.get("lat") if location else None,
                location.get("lon") if location else None
            ))
            db.commit()
            ip_id = cursor.lastrowid

        response = ask_bot(message)

        cursor.execute("""
            INSERT INTO chat_logs
            (session_id, ip_id, user_message, bot_response)
            VALUES (%s, %s, %s, %s)
        """, (session_id, ip_id, message, response))
        db.commit()

        app_logger.info(f"[CHAT] user: '{message[:70]}...' | bot: '{response[:70]}...'")

        return jsonify({"reply": response})

    except Exception as e:
        app_logger.error(f"[CHAT] Critical error: {str(e)}", exc_info=True)
        return jsonify({"reply": "Server error"}), 500

# ────────────────────────────────────────────────
# Admin decorator
# ────────────────────────────────────────────────
def admin_required(f):
    def wrapper(*args, **kwargs):
        if "admin" not in session:
            return redirect("/admin/login")
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

# ────────────────────────────────────────────────
# Admin routes
# ────────────────────────────────────────────────

@app.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    if request.method == "GET":
        return render_template("admin_register.html")

    try:
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm_password", "").strip()
        secret = request.form.get("secret_key", "").strip()

        if not username or not password:
            return render_template("admin_register.html", error="Username and password required")
        if password != confirm:
            return render_template("admin_register.html", error="Passwords do not match")
        if os.getenv("ADMIN_REGISTRATION_KEY") and secret != os.getenv("ADMIN_REGISTRATION_KEY"):
            return render_template("admin_register.html", error="Invalid registration key")

        db = get_db_connection()
        if not db:
            return render_template("admin_register.html", error="Database unavailable")

        cursor = db.cursor()
        cursor.execute("SELECT id FROM admins WHERE username = %s", (username,))
        if cursor.fetchone():
            return render_template("admin_register.html", error="Username already taken")

        hashed = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO admins (username, password, status, role, created_at)
            VALUES (%s, %s, 'inactive', 'admin', NOW())
        """, (username, hashed))
        db.commit()

        return render_template("admin_register.html", success="Registration submitted. Await approval.")
    except Exception as e:
        app_logger.error(f"Admin register error: {str(e)}")
        return render_template("admin_register.html", error="Server error")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html", error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not username or not password:
        return render_template("admin_login.html", error="Username and password required")

    db = get_db_connection()
    if not db:
        return render_template("admin_login.html", error="Database unavailable")

    cursor = db.cursor()
    cursor.execute("SELECT id, password, status FROM admins WHERE username = %s", (username,))
    admin = cursor.fetchone()

    if not admin:
        return render_template("admin_login.html", error="Invalid credentials")

    aid, hashed_pw, status = admin
    if status != "active":
        return render_template("admin_login.html", error="Account not activated")

    if check_password_hash(hashed_pw, password):
        session["admin"] = aid
        cursor.execute("UPDATE admins SET last_login = NOW() WHERE id = %s", (aid,))
        db.commit()
        return redirect("/admin/dashboard")

    return render_template("admin_login.html", error="Invalid credentials")

@app.route("/admin/logout")
@admin_required
def admin_logout():
    session.pop("admin", None)
    return redirect("/admin/login")

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    db = get_db_connection()
    if not db:
        return render_template("admin_dashboard.html", error="Database unavailable")

    cursor = db.cursor(dictionary=True)

    stats = {}
    cursor.execute("SELECT COUNT(*) as c FROM documents")
    stats['total_documents'] = cursor.fetchone()['c']
    cursor.execute("SELECT COUNT(*) as c FROM chat_logs")
    stats['total_messages'] = cursor.fetchone()['c']

    cursor.execute("""
        SELECT chat_logs.id, chat_logs.created_at, ip_addresses.ip_address,
               ip_addresses.city, ip_addresses.country,
               chat_logs.user_message, chat_logs.bot_response
        FROM chat_logs LEFT JOIN ip_addresses ON chat_logs.ip_id = ip_addresses.id
        ORDER BY chat_logs.created_at DESC LIMIT 10
    """)
    recent_chats = cursor.fetchall()

    cursor.execute("""
        SELECT ip_addresses.ip_address, ip_addresses.city, ip_addresses.country,
               COUNT(*) as message_count
        FROM chat_logs JOIN ip_addresses ON chat_logs.ip_id = ip_addresses.id
        GROUP BY ip_addresses.id ORDER BY message_count DESC LIMIT 5
    """)
    top_ips = cursor.fetchall()

    return render_template("admin_dashboard.html",
                         stats=stats, recent_chats=recent_chats, top_ips=top_ips)

@app.route("/admin/knowledge-base", methods=["GET", "POST"])
@admin_required
def admin_knowledge_base():
    error = success = None
    db = get_db_connection()
    if not db:
        return render_template("admin_knowledge_base.html", error="Database unavailable")

    cursor = db.cursor(dictionary=True)

    if request.method == "POST" and 'pdf' in request.files:
        pdf = request.files['pdf']
        if pdf.filename == '':
            error = "No file selected"
        elif not pdf.filename.lower().endswith('.pdf'):
            error = "Only PDF files allowed"
        else:
            filename = pdf.filename
            path = os.path.join("pdfs", filename)
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(path):
                filename = f"{base}_{counter}{ext}"
                path = os.path.join("pdfs", filename)
                counter += 1

            pdf.save(path)
            cursor.execute("INSERT INTO documents (filename, uploaded_at) VALUES (%s, NOW())", (filename,))
            db.commit()
            doc_id = cursor.lastrowid
            process_pdf(path, doc_id)
            success = f"Uploaded and processed: {filename}"

    cursor.execute("""
        SELECT d.id, d.filename, d.uploaded_at,
               COUNT(dc.id) as chunk_count
        FROM documents d LEFT JOIN document_chunks dc ON d.id = dc.document_id
        GROUP BY d.id ORDER BY d.uploaded_at DESC
    """)
    documents = cursor.fetchall()

    return render_template("admin_knowledge_base.html",
                         documents=documents, error=error, success=success)

@app.route("/admin/knowledge-base/document/<int:doc_id>/chunks")
@admin_required
def view_document_chunks(doc_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT filename FROM documents WHERE id = %s", (doc_id,))
    doc = cursor.fetchone()
    if not doc:
        return "Document not found", 404

    cursor.execute("SELECT chunk_index, content, created_at FROM document_chunks WHERE document_id = %s ORDER BY chunk_index", (doc_id,))
    chunks = cursor.fetchall()

    return render_template("admin_document_chunks.html", document=doc, chunks=chunks, doc_id=doc_id)

@app.route("/admin/knowledge-base/delete/<int:doc_id>", methods=["POST"])
@admin_required
def delete_document(doc_id):
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT filename FROM documents WHERE id = %s", (doc_id,))
        doc = cursor.fetchone()
        if not doc:
            return jsonify({"success": False, "message": "Not found"}), 404

        file_path = os.path.join("pdfs", doc['filename'])

        vector_db, _ = get_vector_db()
        if vector_db:
            try:
                results = vector_db.get(where={"document_id": str(doc_id)})
                if results and results.get('ids'):
                    vector_db.delete(ids=results['ids'])
            except Exception as e:
                rag_logger.error(f"Chroma delete failed: {e}")

        cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        db.commit()

        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({"success": True})

    except Exception as e:
        app_logger.error(f"Delete doc {doc_id} error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/admin/ip-addresses")
@admin_required
def admin_ip_addresses():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT ip_address, country, region, city, latitude, longitude,
               COUNT(chat_logs.id) as message_count,
               MAX(chat_logs.created_at) as last_seen
        FROM ip_addresses LEFT JOIN chat_logs ON ip_addresses.id = chat_logs.ip_id
        GROUP BY ip_addresses.id ORDER BY last_seen DESC
    """)
    ips = cursor.fetchall()
    return render_template("admin_ip_addresses.html", ips=ips)

@app.route("/admin/manage-admins")
@admin_required
def admin_manage_admins():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, username, status, role, created_at, last_login FROM admins ORDER BY created_at DESC")
    admins = cursor.fetchall()
    return render_template("admin_manage_admins.html", admins=admins)

@app.route("/admin/toggle-admin-status", methods=["POST"])
@admin_required
def toggle_admin_status():
    admin_id = request.form.get("admin_id")
    new_status = request.form.get("status")
    if not admin_id or new_status not in ['active', 'inactive']:
        return jsonify({"success": False}), 400

    db = get_db_connection()
    if db:
        cursor = db.cursor()
        cursor.execute("UPDATE admins SET status = %s WHERE id = %s", (new_status, admin_id))
        db.commit()
    return jsonify({"success": True})

@app.route("/admin/view-errors")
@admin_required
def admin_view_errors():
    app_logs = {}
    rag_logs = {}

    log_files = {
        "app.log": "logs/app.log",
        "rag.log": "logs/rag.log"
    }

    for name, path in log_files.items():
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-120:]
                content = "".join(lines) or "(no recent entries)"
        except Exception as e:
            content = f"Cannot read log: {str(e)}"

        if "app" in name:
            app_logs[name] = content
        else:
            rag_logs[name] = content

    return render_template(
        "admin_view_errors.html",
        app_logs=app_logs,
        rag_logs=rag_logs
    )

@app.route("/admin/")
@admin_required
def all_messages():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            ip_addresses.id AS ip_id,
            ip_addresses.ip_address,
            ip_addresses.country,
            ip_addresses.city,
            COUNT(chat_logs.id) AS message_count,
            MAX(chat_logs.created_at) AS last_message_at
        FROM ip_addresses
        LEFT JOIN chat_logs ON ip_addresses.id = chat_logs.ip_id
        GROUP BY ip_addresses.id
        HAVING message_count > 0
        ORDER BY last_message_at DESC
    """)
    unique_users = cursor.fetchall()

    selected_ip_id = request.args.get('ip_id', type=int)
    selected_messages = []
    selected_ip_info = None

    if selected_ip_id:
        cursor.execute("""
            SELECT 
                chat_logs.user_message,
                chat_logs.bot_response,
                chat_logs.created_at
            FROM chat_logs
            WHERE chat_logs.ip_id = %s
            ORDER BY chat_logs.created_at ASC
        """, (selected_ip_id,))
        selected_messages = cursor.fetchall()

        cursor.execute("""
            SELECT ip_address, country, city 
            FROM ip_addresses 
            WHERE id = %s
        """, (selected_ip_id,))
        selected_ip_info = cursor.fetchone()

    cursor.close()
    db.close()

    return render_template(
        "all_messages.html",
        unique_users=unique_users,
        selected_messages=selected_messages,
        selected_ip_info=selected_ip_info,
        selected_ip_id=selected_ip_id
    )

# ────────────────────────────────────────────────
# Error Handlers
# ────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("errors/404.html"), 404

@app.errorhandler(500)
def server_error(e):
    app_logger.error(f"500 error: {str(e)}")
    return render_template("errors/500.html"), 500

@app.errorhandler(403)
def forbidden(e):
    return render_template("errors/403.html"), 403

@app.errorhandler(400)
def bad_request(e):
    return render_template("errors/400.html"), 400

@app.errorhandler(401)
def unauthorized(e):
    return render_template("errors/401.html"), 401

@app.errorhandler(405)
def method_not_allowed(e):
    return render_template("errors/405.html"), 405

# ────────────────────────────────────────────────
# Start server
# ────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)