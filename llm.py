import requests
from openai import OpenAI
from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL

class LLM:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def generate(self, system: str, user: str) -> str:
        if LLM_PROVIDER == "openai":
            if not self.client:
                raise RuntimeError("OPENAI_API_KEY is not configured")
            r = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return r.choices[0].message.content or ""
        r = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
