"""
RAG Retriever Module for University-FineTuned-Assistant.
Performs semantic vector search across ingested lecture notes.
"""

import os
import sys
import json
import yaml
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_yaml_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LectureRetriever:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_yaml_config(config_path)
        self.vector_db_dir = self.config["rag"]["vector_db_dir"]
        self.top_k = self.config["rag"].get("top_k", 3)
        self.emb_model_name = self.config["rag"].get("embedding_model", "all-MiniLM-L6-v2")
        self.use_chroma = False
        self.collection = None
        self.fallback_chunks = []

        self._initialize_store()

    def _initialize_store(self):
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            if os.path.exists(self.vector_db_dir):
                client = chromadb.PersistentClient(path=self.vector_db_dir)
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.emb_model_name
                )
                self.collection = client.get_collection(
                    name="university_lectures",
                    embedding_function=ef
                )
                self.use_chroma = True
        except Exception:
            pass

        # Fallback to local JSON index
        if not self.use_chroma:
            fallback_file = os.path.join(self.vector_db_dir, "lecture_chunks.json")
            if os.path.exists(fallback_file):
                with open(fallback_file, "r", encoding="utf-8") as f:
                    self.fallback_chunks = json.load(f)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        results: List[Dict[str, Any]] = []

        if self.use_chroma and self.collection:
            try:
                res = self.collection.query(
                    query_texts=[query],
                    n_results=k
                )
                if res and res["documents"] and res["documents"][0]:
                    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                        results.append({
                            "text": doc,
                            "source": meta.get("source", "Unknown"),
                            "course": meta.get("course", "Unknown")
                        })
                    return results
            except Exception:
                pass

        # Keyword / token overlap fallback
        if self.fallback_chunks:
            q_terms = set(query.lower().split())
            scored = []
            for chunk in self.fallback_chunks:
                chunk_terms = set(chunk["text"].lower().split())
                score = len(q_terms.intersection(chunk_terms))
                scored.append((score, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, item in scored[:k]:
                results.append({
                    "text": item["text"],
                    "source": item.get("source", "Unknown"),
                    "course": item.get("course", "Unknown")
                })

        return results
