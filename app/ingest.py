from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_DIR, EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data" / "sources.md"

model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("product_knowledge")
text = source.read_text(encoding="utf-8")
chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
collection.upsert(
    ids=[f"src-{i}" for i in range(len(chunks))],
    documents=chunks,
    metadatas=[{"source": source.name, "chunk": i} for i in range(len(chunks))],
    embeddings=embeddings,
)
print(f"Indexed {len(chunks)} chunks")
