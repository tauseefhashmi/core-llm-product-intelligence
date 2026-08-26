from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import Retriever
from app.llm import LLM

app = FastAPI(title="LLM Product Intelligence")
retriever = Retriever()
llm = LLM()

class AskRequest(BaseModel):
    question: str
    k: int = 4

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    context = retriever.search(req.question, req.k)
    packed = "\n\n".join(f"[Source {i+1}] {c['text']}" for i, c in enumerate(context))
    prompt = f"Answer only from the supplied context. If the evidence is insufficient, say so. Cite sources as [Source N].\n\nContext:\n{packed}\n\nQuestion: {req.question}"
    answer = llm.generate("You are a careful product research analyst.", prompt)
    return {"answer": answer, "sources": context}
