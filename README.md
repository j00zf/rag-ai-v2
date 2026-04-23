# College Chatbot System 

A Flask-based AI chatbot system that answers user queries using uploaded college documents (PDFs) with Retrieval-Augmented Generation (RAG).

Author: Joseph Sebastian

##  Features

- AI Chatbot powered by Groq (LLaMA 3)
- Upload PDFs to build knowledge base
- Semantic search using ChromaDB + embeddings
- Context-aware answers (no hallucination)
- Admin panel for:
- Document management
- FAQ management (CRUD)
- Chat logs & analytics
- IP tracking with geolocation
- Server monitoring
- Admin authentication system
- Dashboard with usage stats

---

##  Tech Stack

- **Backend:** Flask (Python)
- **Database:** MySQL
- **Vector DB:** Chroma
- **Embeddings:** HuggingFace (MiniLM)
- **LLM API:** Groq (LLaMA 3)
- **Frontend:** HTML + Jinja2
- **Other:** LangChain, psutil

---

## 📂 Project Structure
project/
│
├── pdfs/ # Uploaded documents
├── chroma_db/ # Vector database
├── logs/ # Log files
│
├── templates/ # HTML templates
├── static/ # Static files
│
├── app.py # Main Flask app
├── .env # Environment variables
└── README.md

---
## How to  Run

### Step 1: Clone 
```bash
git clone <your-repo-url>
cd project
```

### Step 2: Create python env  
```bash
python -m venv venv
```
Activate:
```bash
#windows
venv\Scripts\activate
#Linux/Mac
source venv/bin/activate
```
### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```
### Step 4: Create the .env file with following
SECRET_KEY=your_secret_key

MYSQL_HOST=127.0.0.1
MYSQL_PORT=*sql_port*
MYSQL_USER=*sql_user*
MYSQL_PASSWORD=*sql_pass*
MYSQL_DB=*db_name*

GROQ_API_KEY=*your_groq_api_key*
IPGEOLOCATION_API_KEY=*your_ipgeo_key*

### Step: 5 Databse
Create the database with 
```sql
CREATE DATABASE db_name;
```
By importing **dbschema.sql** you can create the database schema

### Step 6: Run App
```bash
python app.py
```
**Open in Browser**
Main App → http://localhost:5000
Admin Login → http://localhost:5000/admin/login

*For Admin Setup go to* http://localhost:5000/admin/

### *Error Loging*
the error files are dynamically generated when the app gets excuted for the first time itself. The following are the error logging files
- logs/app.log → stores app activity
- logs/rag.log → stores AI + vector errors

## Working of the App

1. Upload PDF via admin panel
2. PDF is split into chunks
3. the data is Stored in:
    - MySQL (for tracking)
    - ChromaDB (for semantic search)
4. User asks question
5. Relevant chunks retrieved and sent to LLM  for generating Answers

