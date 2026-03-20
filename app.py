import os
import uuid
import logging
from gotrue import datetime
import requests
import time
import traceback
import psutil
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, redirect, session, flash
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
            port=int(os.getenv("MYSQL_PORT", 3307)),           # ← updated default port
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DB", "ollege_chatbot"),  # ← matches your DB name
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
            search_kwargs={"k": 8, "score_threshold": 0.24}   # ← tuned a bit higher k, lower threshold
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
# PDF processing – IMPROVED with description in metadata
# ────────────────────────────────────────────────
def process_pdf(path, document_id, description=""):
    app_logger.info(f"[PDF] Processing file: {path} (desc: {description[:60]}...)")
    vector_db, _ = get_vector_db()
    
    if vector_db is None:
        rag_logger.error("Vector DB not initialized – cannot process PDF")
        return

    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        # Save chunks to MySQL (unchanged)
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

        # Prepare rich metadata – description goes here!
        metadata_list = [
            {
                "document_id":   str(document_id),
                "chunk_index":   i,
                "source":        os.path.basename(path),
                "description":   description.strip() if description else "No description provided",
            }
            for i in range(len(docs))
        ]

        # Attach metadata
        for doc, meta in zip(docs, metadata_list):
            doc.metadata.update(meta)

        # Add to vector store
        vector_db.add_documents(docs)
        vector_db.persist()

        new_count = vector_db._collection.count()
        app_logger.info(f"[PDF] Added {len(docs)} chunks → total now: {new_count}")

    except Exception as e:
        rag_logger.error(f"[PDF] Processing failed: {str(e)}", exc_info=True)
        app_logger.error(f"[PDF] Processing failed: {str(e)}", exc_info=True)

# ────────────────────────────────────────────────
# RAG – now uses document descriptions strongly
# ────────────────────────────────────────────────
def ask_bot(question):
    _, retriever = get_vector_db()
    if not retriever or not groq_client:
        return "The AI system is temporarily unavailable."

    try:
        docs = retriever.invoke(question)
        if not docs:
            return "I don't have information about that in the college documents."

        # Build rich context with document descriptions
        context_parts = []
        doc_descriptions = set()  # avoid duplicates

        for doc in docs:
            meta = doc.metadata
            src  = meta.get("source", "unknown.pdf")
            desc = meta.get("description", "No description").strip()

            if desc and desc != "No description provided":
                doc_descriptions.add(f"• {src} — {desc}")

            header = f"Document: {src}"
            if desc and desc != "No description provided":
                header += f"\nPurpose: {desc}"

            context_parts.append(f"{header}\n{doc.page_content}")

        context = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(context_parts)

        # List of all relevant document purposes in prompt
        available_docs_str = "\n".join(sorted(doc_descriptions)) if doc_descriptions else "(no descriptions available)"

        prompt = f"""You are a helpful college information assistant.Anser based on the uploded documents. and desciptions. 
Do NOT use your general knowledge. Do NOT make up information.
If nothing relevant is found, reply exactly:
"I don't have information about that in the available college documents."

Available documents and their purpose:
{available_docs_str}

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
# IP Helpers (unchanged)
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
# Security headers to allow iframe embedding 
# (for embedding the chatbot in college portals)
# ────────────────────────────────────────────────

@app.after_request
def allow_iframe(response):
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Content-Security-Policy"] = "frame-ancestors *"

    # CORS (important for cross-site)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"

    return response
# ────────────────────────────────────────────────
# Public routes (chat mostly unchanged)
# ────────────────────────────────────────────────
@app.route("/")
def home():
    is_logged_in = 'user_id' in session  # or however you check auth
    return render_template('chat.html', is_logged_in=is_logged_in)

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
    # CPU usage (percentage over last second)
    cpu_percent = psutil.cpu_percent(interval=1)

    # RAM / Memory
    mem = psutil.virtual_memory()
    memory_total_gb = round(mem.total / (1024 ** 3), 2)
    memory_used_gb = round(mem.used / (1024 ** 3), 2)
    memory_free_gb = round(mem.free / (1024 ** 3), 2)
    memory_percent = mem.percent

    # Disk usage (root partition /)
    disk = psutil.disk_usage('/')
    disk_total_gb = round(disk.total / (1024 ** 3), 2)
    disk_used_gb = round(disk.used / (1024 ** 3), 2)
    disk_free_gb = round(disk.free / (1024 ** 3), 2)
    disk_percent = disk.percent

    # Load average (1, 5, 15 min)
    load_avg = psutil.getloadavg()
    load_1min, load_5min, load_15min = [round(x, 2) for x in load_avg]

    # Uptime
    uptime_seconds = time.time() - psutil.boot_time()
    uptime_days = int(uptime_seconds // 86400)
    uptime_hours = int((uptime_seconds % 86400) // 3600)
    uptime_str = f"{uptime_days} days, {uptime_hours} hours"

   

    db = get_db_connection()
    if not db:
        return render_template("admin_dashboard.html", error="Database unavailable")

    cursor = db.cursor(dictionary=True)

    stats = {
        "cpu_percent": cpu_percent,
        
        "memory_percent": memory_percent,
        
    }
    cursor.execute("SELECT COUNT(*) as c FROM documents")
    stats['total_documents'] = cursor.fetchone()['c']
    cursor.execute("SELECT COUNT(*) as c FROM chat_logs")
    stats['total_messages'] = cursor.fetchone()['c']
    cursor.execute("SELECT COUNT(DISTINCT ip_id) as c FROM chat_logs")
    stats['total_users'] = cursor.fetchone()['c']

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
        description = request.form.get('description', '').strip()  # new field

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

            # Insert with description
            cursor.execute(
                "INSERT INTO documents (filename, description, uploaded_at) VALUES (%s, %s, NOW())",
                (filename, description)
            )
            db.commit()
            doc_id = cursor.lastrowid

            process_pdf(path, doc_id)
            success = f"Uploaded and processed: {filename}"

    # Fetch documents including description
    cursor.execute("""
        SELECT d.id, d.filename, d.description, d.uploaded_at,
               COUNT(dc.id) as chunk_count
        FROM documents d 
        LEFT JOIN document_chunks dc ON d.id = dc.document_id
        GROUP BY d.id 
        ORDER BY d.uploaded_at DESC
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
        if not db:
            return jsonify({"success": False, "message": "Database unavailable"}), 503

        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT filename FROM documents WHERE id = %s", (doc_id,))
        doc = cursor.fetchone()
        if not doc:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "message": "Document not found"}), 404
            flash("Document not found", "error")
            return redirect("/admin/knowledge-base")

        file_path = os.path.join("pdfs", doc['filename'])

        # Delete from Chroma
        vector_db, _ = get_vector_db()
        if vector_db:
            try:
                results = vector_db.get(where={"document_id": str(doc_id)})
                if results and results.get('ids'):
                    vector_db.delete(ids=results['ids'])
            except Exception as e:
                rag_logger.error(f"Chroma delete failed: {e}")

        # Delete from MySQL
        cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        db.commit()

        if os.path.exists(file_path):
            os.remove(file_path)

        message = f"Document '{doc['filename']}' deleted successfully"

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": True, "message": message})
        else:
            flash(message, "success")
            return redirect("/admin/knowledge-base")

    except Exception as e:
        app_logger.error(f"Delete doc {doc_id} error: {str(e)}", exc_info=True)
        error_msg = f"Error deleting document: {str(e)}"

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "message": error_msg}), 500
        else:
            flash(error_msg, "error")
            return redirect("/admin/knowledge-base")
        
@app.route("/admin/ip-addresses")
@admin_required
def admin_ip_addresses():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT ip_address, country, region, city, latitude, longitude,
               COUNT(chat_logs.id) as message_count,
               MAX(chat_logs.created_at) as last_seen
        FROM ip_addresses 
        LEFT JOIN chat_logs ON ip_addresses.id = chat_logs.ip_id
        GROUP BY ip_addresses.id 
        ORDER BY last_seen DESC
    """)
    ips = cursor.fetchall()
    map_data = []
    for ip in ips:
        if ip["latitude"] and ip["longitude"]:
            map_data.append({
                "ip": ip["ip_address"],
                "city": ip["city"] or "Unknown",
                "country": ip["country"] or "",
                "lat": float(ip["latitude"]),
                "lng": float(ip["longitude"]),
                "messages": ip["message_count"],
                "last_seen": str(ip["last_seen"]) if ip["last_seen"] else "Never"
            })

    import json

    return render_template(
        "admin_ip_addresses.html",
        ips=ips,
        map_data_json=json.dumps(map_data)  
    )

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

@app.route("/admin/all-user-messages")
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

@app.route("/admin/server-status")
@admin_required
def server_status():
    # CPU usage (percentage over last second)
    cpu_percent = psutil.cpu_percent(interval=1)

    # RAM / Memory
    mem = psutil.virtual_memory()
    memory_total_gb = round(mem.total / (1024 ** 3), 2)
    memory_used_gb = round(mem.used / (1024 ** 3), 2)
    memory_free_gb = round(mem.free / (1024 ** 3), 2)
    memory_percent = mem.percent

    # Disk usage (root partition /)
    disk = psutil.disk_usage('/')
    disk_total_gb = round(disk.total / (1024 ** 3), 2)
    disk_used_gb = round(disk.used / (1024 ** 3), 2)
    disk_free_gb = round(disk.free / (1024 ** 3), 2)
    disk_percent = disk.percent

    # Load average (1, 5, 15 min)
    load_avg = psutil.getloadavg()
    load_1min, load_5min, load_15min = [round(x, 2) for x in load_avg]

    # Uptime
    uptime_seconds = time.time() - psutil.boot_time()
    uptime_days = int(uptime_seconds // 86400)
    uptime_hours = int((uptime_seconds % 86400) // 3600)
    uptime_str = f"{uptime_days} days, {uptime_hours} hours"

    stats = {
        "cpu_percent": cpu_percent,
        "memory": {
            "total_gb": memory_total_gb,
            "used_gb": memory_used_gb,
            "free_gb": memory_free_gb,
            "percent": memory_percent
        },
        "disk": {
            "total_gb": disk_total_gb,
            "used_gb": disk_used_gb,
            "free_gb": disk_free_gb,
            "percent": disk_percent
        },
        "load_average": {
            "1min": load_1min,
            "5min": load_5min,
            "15min": load_15min
        },
        "uptime": uptime_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    }

    return render_template("admin_server_status.html", stats=stats)

@app.route("/admin/db")   # ← nicer, shorter URL: /admin/db
@admin_required
def admin_db():
    return app.send_static_file(
        "db-admin-tool",                  # ← use the new filename
        mimetype="text/html",             # ← force browser to treat it as HTML
        conditional=True                  # ← optional: better caching
    )

#────────────────────────────────────────────────
# FAQ Management - Full CRUD + Toggle in one route
# ────────────────────────────────────────────────
# CRUD Helpers (already improved from previous)
# ────────────────────────────────────────────────

def get_all_faqs():
    db = get_db_connection()
    if not db: return []
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, question, answer, category, status, created_at, updated_at
            FROM faqs
            ORDER BY 
                CASE WHEN status = 'active' THEN 0 ELSE 1 END,
                question ASC
        """)
        return cursor.fetchall()
    except Exception as e:
        app_logger.error(f"get_all_faqs error: {str(e)}")
        return []
    finally:
        if 'cursor' in locals(): cursor.close()


def add_faq(question: str, answer: str, category: str | None = None) -> tuple[bool, str]:
    db = get_db_connection()
    if not db: return False, "DB connection failed"

    question = (question or "").strip()
    answer   = (answer   or "").strip()
    category = (category or "").strip() or None

    if not question or not answer:
        return False, "Question and answer required"

    try:
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO faqs 
            (question, answer, category, status, created_at, updated_at)
            VALUES (%s, %s, %s, 'active', NOW(), NOW())
        """, (question, answer, category))
        db.commit()
        return True, "FAQ added"
    except Exception as e:
        app_logger.error(f"add_faq error: {str(e)}")
        return False, "Error adding FAQ"
    finally:
        if 'cursor' in locals(): cursor.close()


def update_faq(faq_id: int, 
               question: str | None = None,
               answer: str | None = None,
               category: str | None = None,
               status: str | None = None) -> tuple[bool, str]:
    db = get_db_connection()
    if not db: return False, "DB connection failed"

    updates = []
    params = []

    if question is not None:
        q = (question or "").strip()
        if not q: return False, "Question cannot be empty"
        updates.append("question = %s")
        params.append(q)

    if answer is not None:
        a = (answer or "").strip()
        if not a: return False, "Answer cannot be empty"
        updates.append("answer = %s")
        params.append(a)

    if category is not None:
        cat = (category or "").strip() or None
        updates.append("category = %s")
        params.append(cat)

    if status in ('active', 'inactive'):
        updates.append("status = %s")
        params.append(status)

    if not updates:
        return False, "No fields to update"

    params.append(faq_id)

    try:
        cursor = db.cursor()
        cursor.execute(f"""
            UPDATE faqs
            SET {', '.join(updates)}, updated_at = NOW()
            WHERE id = %s
        """, params)
        db.commit()
        return cursor.rowcount > 0, "FAQ updated" if cursor.rowcount > 0 else "FAQ not found"
    except Exception as e:
        app_logger.error(f"update_faq error: {str(e)}")
        return False, "Error updating FAQ"
    finally:
        if 'cursor' in locals(): cursor.close()


def toggle_faq_status(faq_id: int) -> tuple[bool, str]:
    db = get_db_connection()
    if not db: return False, "DB connection failed"

    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT status FROM faqs WHERE id = %s", (faq_id,))
        row = cursor.fetchone()
        if not row: return False, "FAQ not found"

        new_status = 'inactive' if row['status'] == 'active' else 'active'

        cursor.execute("UPDATE faqs SET status = %s, updated_at = NOW() WHERE id = %s", 
                       (new_status, faq_id))
        db.commit()
        return True, f"Status → {new_status}"
    except Exception as e:
        app_logger.error(f"toggle_faq_status error: {str(e)}")
        return False, "Error toggling status"
    finally:
        if 'cursor' in locals(): cursor.close()


def delete_faq(faq_id: int) -> tuple[bool, str]:
    db = get_db_connection()
    if not db: return False, "DB connection failed"

    try:
        cursor = db.cursor()
        cursor.execute("DELETE FROM faqs WHERE id = %s", (faq_id,))
        db.commit()
        return cursor.rowcount > 0, "FAQ deleted" if cursor.rowcount > 0 else "FAQ not found"
    except Exception as e:
        app_logger.error(f"delete_faq error: {str(e)}")
        return False, "Error deleting FAQ"
    finally:
        if 'cursor' in locals(): cursor.close()


# ────────────────────────────────────────────────
# Single Admin Route — handles ALL operations
# ────────────────────────────────────────────────

@app.route("/admin/manage-faq", methods=["GET", "POST"])
@admin_required
def manage_faq():
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json
    message = None
    message_type = "info"

    if request.method == "POST":
        action = request.form.get("action")
        faq_id = request.form.get("faq_id")

        # ── ADD ───────────────────────────────────────
        if action == "add":
            success, msg = add_faq(
                request.form.get("question", "").strip(),
                request.form.get("answer",   "").strip(),
                request.form.get("category", "").strip() or None
            )
            if is_ajax:
                return jsonify({"success": success, "message": msg}), 200 if success else 400
            message = msg
            message_type = "success" if success else "danger"

        # ── EDIT ──────────────────────────────────────
        elif action == "edit":
            if not faq_id or not faq_id.isdigit():
                msg = "Invalid FAQ ID"
            else:
                success, msg = update_faq(
                    int(faq_id),
                    request.form.get("question"),
                    request.form.get("answer"),
                    request.form.get("category", "").strip() or None,
                    request.form.get("status")  # optional - usually not changed via edit
                )
            if is_ajax:
                return jsonify({"success": success, "message": msg}), 200 if success else 400
            message = msg
            message_type = "success" if success else "danger"

        # ── TOGGLE ────────────────────────────────────
        elif action == "toggle":
            if not faq_id or not faq_id.isdigit():
                msg = "Invalid FAQ ID"
            else:
                success, msg = toggle_faq_status(int(faq_id))
            if is_ajax:
                return jsonify({"success": success, "message": msg}), 200 if success else 400
            message = msg
            message_type = "success" if success else "danger"

        # ── DELETE ────────────────────────────────────
        elif action == "delete":
            if not faq_id or not faq_id.isdigit():
                msg = "Invalid FAQ ID"
            else:
                success, msg = delete_faq(int(faq_id))
            if is_ajax:
                return jsonify({"success": success, "message": msg}), 200 if success else 400
            message = msg
            message_type = "success" if success else "danger"

    faqs = get_all_faqs()

    return render_template(
        "admin_manage_faq.html",
        faqs=faqs,
        success=message if message_type == "success" else None,
        error=message if message_type in ("danger", "error") else None,
        message=message,
        message_type=message_type
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
