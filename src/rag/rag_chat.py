"""
RAG-Augmented University Assistant.
Combines Vector Retrieval over lecture notes with the Fine-Tuned LLM
to answer lecture-specific inquiries with source citations.
"""

import os
import sys
import argparse
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.model_loader import UniversityAssistant
from src.rag.retriever import LectureRetriever


class RAGUniversityAssistant:
    def __init__(self, config_path: str = "configs/config.yaml", model_type: str = "lora"):
        self.retriever = LectureRetriever(config_path=config_path)
        self.assistant = UniversityAssistant(model_type=model_type, config_path=config_path)

    def answer_with_rag(self, query: str, top_k: int = 3) -> Tuple[str, list]:
        """Retrieve relevant lecture context, construct grounded prompt, and generate answer."""
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        if not retrieved_docs:
            # Fallback to standard model response if no docs found
            response = self.assistant.generate_response(user_message=query)
            return response, []

        context_str = "\n\n".join([
            f"[Source: {d['source']} | Course: {d['course']}]\n{d['text']}"
            for d in retrieved_docs
        ])

        augmented_system_prompt = (
            "You are the official University Academic & Lecture Assistant for Apex National University. "
            "Use the provided lecture notes and course excerpts below to answer the student's question accurately. "
            "Always cite the source course or lecture when referencing specific facts.\n\n"
            f"--- RELEVANT LECTURE EXCERPTS ---\n{context_str}\n----------------------------------"
        )

        augmented_user_query = f"Question: {query}\n\nPlease provide a clear, pedagogical answer based on the lecture material."

        response = self.assistant.generate_response(
            user_message=augmented_user_query,
            system_prompt=augmented_system_prompt,
            max_new_tokens=250,
            temperature=0.2
        )

        sources = [f"{d['course']} ({d['source']})" for d in retrieved_docs]
        return response, sources


def main():
    parser = argparse.ArgumentParser(description="University RAG Lecture Assistant CLI")
    parser.add_argument("--model", type=str, default="lora", choices=["lora", "base"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    print("=" * 65)
    print("    APEX NATIONAL UNIVERSITY - LECTURE RAG ASSISTANT")
    print("=" * 65)
    print("Ask questions about lecture notes (e.g. SOLID principles, ACID, TCP/IP).")
    print("Type 'exit' to quit.\n")

    rag_bot = RAGUniversityAssistant(config_path=args.config, model_type=args.model)

    while True:
        try:
            query = input("\nStudent Query: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nAssistant: Goodbye! Best of luck with your studies.")
                break

            response, sources = rag_bot.answer_with_rag(query)
            print(f"\nAssistant:\n{response}")
            if sources:
                print("\n📚 Referenced Lecture Sources:")
                for s in set(sources):
                    print(f"  - {s}")

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"[!] Error: {e}")


if __name__ == "__main__":
    main()
