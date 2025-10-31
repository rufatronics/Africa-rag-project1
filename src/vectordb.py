import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from datasets import load_dataset

class VectorDB:
    def __init__(self):
        self.client = chromadb.Client()
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            "africa_rag", embedding_function=self.embedding_model
        )

    def chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        for i, doc in enumerate(documents):
            chunks = self.chunk_text(doc["content"])
            ids = [f"chunk_{i}_{j}" for j in range(len(chunks))]
            self.collection.add(
                documents=chunks,
                ids=ids,
                metadatas=[doc.get("metadata", {})] * len(chunks)
            )

    def has_data(self) -> bool:
        try:
            count = len(self.collection.get()["ids"])
            return count > 0
        except:
            return False

    def ingest_african_datasets(self):
        datasets_to_use = [
            ("masakhane/masakhane-ner", "train", "tokens"),
            ("joelito/AfricanStories", "train", "story"),
            ("Kelechi/Africa-News-Dataset", "train", "text")
        ]
        documents = []
        for name, split, key in datasets_to_use:
            try:
                data = load_dataset(name, split=split)
                for item in data.select(range(min(50, len(data)))):
                    content = " ".join(item[key]) if isinstance(item[key], list) else str(item[key])
                    documents.append({"content": content, "metadata": {"source": name}})
            except Exception as e:
                print(f"⚠️ Skipped {name}: {e}")
        self.add_documents(documents)

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "ids": results["ids"][0]
        }

    # Alias for compatibility
    def similarity_search(self, query: str, k: int = 3):
        return self.search(query, n_results=k)
