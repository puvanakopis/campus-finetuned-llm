"""
RAG Ingestion Pipeline for University-FineTuned-Assistant.
Processes course lecture notes, chunks text, generates embeddings,
and builds a local vector store.
"""

import os
import sys
import glob
import json
import yaml
from typing import List, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_yaml_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    tokens = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i:i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks


def extract_lecture_files(lectures_dir: str) -> List[Dict[str, str]]:
    """Recursively load all text and markdown files from lectures directory."""
    documents = []
    pattern = os.path.join(lectures_dir, "**", "*.*")
    files = glob.glob(pattern, recursive=True)

    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            rel_path = os.path.relpath(file_path, lectures_dir)
            course = os.path.dirname(rel_path) or "General"
            
            documents.append({
                "source": rel_path,
                "course": course,
                "content": content
            })
        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                content = " ".join([page.extract_text() or "" for page in reader.pages])
                rel_path = os.path.relpath(file_path, lectures_dir)
                course = os.path.dirname(rel_path) or "General"
                documents.append({
                    "source": rel_path,
                    "course": course,
                    "content": content
                })
            except Exception as e:
                print(f"[!] Warning: Could not read PDF '{file_path}': {e}")

    return documents


def build_vector_store(config_path: str = "configs/config.yaml"):
    config = load_yaml_config(config_path)
    lectures_dir = config["rag"]["lectures_dir"]
    vector_db_dir = config["rag"]["vector_db_dir"]
    chunk_size = config["rag"].get("chunk_size", 400)
    overlap = config["rag"].get("chunk_overlap", 50)
    emb_model_name = config["rag"].get("embedding_model", "all-MiniLM-L6-v2")

    print("=" * 60)
    print("           LECTURE NOTES RAG INGESTION PIPELINE")
    print("=" * 60)
    print(f"[*] Scanning lectures from: '{lectures_dir}'...")

    docs = extract_lecture_files(lectures_dir)
    print(f"[*] Found {len(docs)} lecture documents.")

    # Prepare chunks
    all_chunks: List[Dict] = []
    chunk_id = 0
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            all_chunks.append({
                "id": f"chunk_{chunk_id}",
                "source": doc["source"],
                "course": doc["course"],
                "text": c
            })
            chunk_id += 1

    print(f"[*] Total processed chunks: {len(all_chunks)}")

    # Try ChromaDB if installed, otherwise build lightweight local index
    os.makedirs(vector_db_dir, exist_ok=True)
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        print(f"[*] Initializing ChromaDB vector store at '{vector_db_dir}'...")
        client = chromadb.PersistentClient(path=vector_db_dir)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=emb_model_name
        )

        collection = client.get_or_create_collection(
            name="university_lectures",
            embedding_function=sentence_transformer_ef
        )

        # Ingest chunks
        ids = [c["id"] for c in all_chunks]
        documents = [c["text"] for c in all_chunks]
        metadatas = [{"source": c["source"], "course": c["course"]} for c in all_chunks]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[+] Successfully ingested {len(all_chunks)} chunks into ChromaDB.")

    except Exception as e:
        print(f"[*] Building local JSON vector metadata store (ChromaDB notice: {e})...")
        fallback_file = os.path.join(vector_db_dir, "lecture_chunks.json")
        with open(fallback_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)
        print(f"[+] Local fallback index saved to: {fallback_file}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    build_vector_store()
