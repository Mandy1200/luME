import os
import sqlite3
import json
import math
from typing import List, Dict, Any, Tuple
from lume.debate import get_client

DB_NAME = "lume_semantic_memory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding TEXT,
            last_modified REAL,
            UNIQUE(filepath, chunk_index)
        )
    """)
    conn.commit()
    conn.close()

def get_embedding(text: str, model_name: str = "nomic-embed-text") -> List[float]:
    """
    Fetches embedding for a given text using Ollama or OpenAI.
    Defaults to 'nomic-embed-text' (Ollama default) or 'text-embedding-3-small' if using OpenAI.
    """
    client = get_client()
    # If the user is using OpenAI API key, swap the default embedding model
    if os.environ.get("OPENAI_API_KEY") and model_name == "nomic-embed-text":
        model_name = "text-embedding-3-small"
        
    try:
        response = client.embeddings.create(
            input=[text],
            model=model_name
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Failed to get embedding from {model_name}: {e}. Returning zero vector.")
        return [0.0] * 384  # Return a dummy vector so system doesn't crash

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(a * a for a in v2))
    if magnitude_v1 == 0.0 or magnitude_v2 == 0.0:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def index_file(filepath: str, embedding_model: str = "nomic-embed-text"):
    """
    Reads a file, chunks it, embeds each chunk, and saves it to SQLite database.
    """
    if not os.path.exists(filepath):
        return
        
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ Could not read file {filepath}: {e}")
        return
        
    last_mod = os.path.getmtime(filepath)
    chunks = chunk_text(content)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Delete old chunks of this file
    cursor.execute("DELETE FROM file_chunks WHERE filepath = ?", (filepath,))
    
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        print(f"🧠 Embedding chunk {idx} of {os.path.basename(filepath)}...")
        vector = get_embedding(chunk, embedding_model)
        vector_json = json.dumps(vector)
        
        cursor.execute("""
            INSERT OR REPLACE INTO file_chunks (filepath, chunk_index, content, embedding, last_modified)
            VALUES (?, ?, ?, ?, ?)
        """, (filepath, idx, chunk, vector_json, last_mod))
        
    conn.commit()
    conn.close()
    print(f"✅ Successfully indexed {filepath}")

def remove_file_from_index(filepath: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM file_chunks WHERE filepath = ?", (filepath,))
    conn.commit()
    conn.close()
    print(f"🗑️ Removed {filepath} from index")

def semantic_search(query: str, top_k: int = 3, embedding_model: str = "nomic-embed-text") -> List[Dict[str, Any]]:
    """
    Searches index for similar chunks and returns the top_k hits.
    """
    init_db()
    query_vector = get_embedding(query, embedding_model)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT filepath, chunk_index, content, embedding FROM file_chunks")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for filepath, chunk_index, content, embedding_json in rows:
        embedding = json.loads(embedding_json)
        sim = cosine_similarity(query_vector, embedding)
        results.append({
            "filepath": filepath,
            "chunk_index": chunk_index,
            "content": content,
            "similarity": sim
        })
        
    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# Directory Watcher for background syncing
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentWatcherHandler(FileSystemEventHandler):
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.embedding_model = embedding_model
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt', '.md', '.py', '.json', '.pdf')):
            print(f"📝 File modified: {event.src_path}")
            index_file(event.src_path, self.embedding_model)
            
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt', '.md', '.py', '.json', '.pdf')):
            print(f"🆕 File created: {event.src_path}")
            index_file(event.src_path, self.embedding_model)
            
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt', '.md', '.py', '.json', '.pdf')):
            print(f"❌ File deleted: {event.src_path}")
            remove_file_from_index(event.src_path)

def start_watcher(directory_path: str, embedding_model: str = "nomic-embed-text") -> Observer:
    """
    Starts watching the directory in the background.
    """
    init_db()
    # First, index existing files
    print(f"📂 Initializing index of directory: {directory_path}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.txt', '.md', '.py', '.json')):
                full_path = os.path.join(root, file)
                index_file(full_path, embedding_model)
                
    event_handler = DocumentWatcherHandler(embedding_model)
    observer = Observer()
    observer.schedule(event_handler, path=directory_path, recursive=True)
    observer.start()
    print(f"👀 Watcher started on: {directory_path}")
    return observer
