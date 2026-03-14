from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embeddings OK")
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    print("Chroma OK, count:", db._collection.count())
except Exception as e:
    print("Error:", str(e))