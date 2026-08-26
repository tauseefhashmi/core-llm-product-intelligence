import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_DIR, EMBEDDING_MODEL

class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection("product_knowledge")

    def search(self, query: str, k: int = 4):
        q = self.embedder.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=q, n_results=k)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        return [{"text": d, "source": m.get("source", "unknown"), "chunk": m.get("chunk")} for d, m in zip(docs, metas)]
