# LLM Product Intelligence

A portfolio-grade LLM engineering project inspired by the **AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents**.

## What it demonstrates

- Local + frontier LLM abstraction (Ollama/OpenAI)
- Embeddings + vector retrieval with Chroma
- RAG with citations back to source documents
- Structured outputs with Pydantic
- Product recommendation / deal analysis agent with tools
- Evaluation hooks for retrieval and answer quality
- QLoRA fine-tuning training pipeline (optional GPU/Colab)
- FastAPI API + Streamlit UI
- Docker Compose for reproducible local execution

## Architecture

```text
                  +----------------------+
                  |      Streamlit       |
                  |  Chat / Deal Search   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |       FastAPI         |
                  | RAG + Agent endpoints |
                  +----+-------------+----+
                       |             |
                +------v----+   +----v------+
                |  Chroma   |   | LLM Router |
                |  VectorDB |   | Ollama/API  |
                +-----------+   +-----+-------+
                                      |
                              +-------v-------+
                              | Tool-calling  |
                              | deal analyzer |
                              +---------------+

        Optional training path:
        product text -> SFT dataset -> QLoRA -> Hugging Face adapter
```

## Quick start

### 1. Local Python

```powershell
uv venv
uv pip install -r requirements.txt
copy .env.example .env
python -m app.ingest
uv run uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
uv run streamlit run app/ui.py --server.port 8501
```

### 2. Docker

```powershell
docker compose up --build
```

Then open `http://localhost:8501`.

## Example questions

- "Which laptops under €1,000 have the best battery-life evidence?"
- "Compare Product A and Product B using only the loaded sources."
- "Find deals with strong value signals but avoid refurbished products."

## QLoRA

The `training/` folder contains a reusable supervised fine-tuning pipeline. It is intentionally separated from inference so the application still runs without a GPU.

```powershell
python training/prepare_dataset.py
# Then train on Colab / CUDA:
python training/train_qlora.py
```

The fine-tuned adapter can be loaded through the same model-router abstraction later.

## Portfolio talking points

1. Why RAG instead of immediately fine-tuning?
2. How do retrieval quality and answer quality differ?
3. When should a small local model be preferred over a frontier API?
4. What belongs in the prompt vs. the vector store vs. fine-tuning?
5. How would you productionize this with evaluation, tracing, caching and model fallback?
