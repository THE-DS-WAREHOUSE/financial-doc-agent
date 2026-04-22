# financial-doc-agent

A LangGraph-powered multi-agent system for financial document Q&A — uploads a PDF, chunks it using multiple strategies, embeds it with HuggingFace, retrieves relevant context from ChromaDB, and returns a structured answer validated by Pydantic AI. Every run is traced and evaluated in LangSmith.

---

## Project Structure

```
financial-doc-agent/
├── app/
│   ├── app.py              # FastAPI endpoints and lifespan
│   ├── agent.py            # LangGraph agent — 4 nodes, state machine
│   ├── chunking.py         # Fixed, recursive, and semantic chunking strategies
│   ├── embeddings.py       # HuggingFace embedding model (all-MiniLM-L6-v2)
│   ├── retrieval.py        # ChromaDB vector store — store and retrieve chunks
│   ├── evaluation.py       # LangSmith evaluation — faithfulness, relevance, correctness
│   └── schemas.py          # Pydantic AI input/output schemas
├── docs/                   # Financial PDFs for testing
├── evals/                  # LangSmith evaluation datasets
├── create_test_pdf.py      # Script to generate a sample financial PDF
├── Dockerfile              # Containerizes the FastAPI app
├── docker-compose.yml      # Orchestrates API + ChromaDB volume
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not committed)
```

---

## Features

- **LangGraph agent** with 4 sequential nodes — ingest, retrieval, reasoning, structured output
- **3 chunking strategies** — fixed-size, recursive, and semantic
- **HuggingFace embeddings** — free, runs locally, no API key needed
- **ChromaDB** — local vector database with persistent storage
- **Pydantic AI** — validates and structures every agent response
- **LangSmith tracing** — every node execution is logged with latency, tokens, and cost
- **LangSmith evaluation** — faithfulness, relevance, and correctness evaluators
- **FastAPI** — REST interface with Swagger UI
- **Docker** — fully containerized, runs anywhere

---

## Tech Stack

- `LangGraph` — stateful agent orchestration
- `Pydantic AI` — structured output validation
- `ChromaDB` — free local vector database
- `HuggingFace` — sentence-transformers/all-MiniLM-L6-v2 embeddings
- `LangSmith` — tracing, observability, and evaluation
- `FastAPI` — REST API framework
- `OpenAI GPT-4o-mini` — LLM backbone
- `Docker Compose` — containerization

---

## Agent Architecture

```
[ PDF Upload ]
      ↓
[ Node 1: Ingestion ]
  └── Chunks document using selected strategy (fixed / recursive / semantic)
  └── Stores chunks in ChromaDB
      ↓
[ Node 2: Retrieval ]
  └── Embeds the question with HuggingFace
  └── Retrieves top-k most relevant chunks from ChromaDB
      ↓
[ Node 3: Reasoning ]
  └── Sends retrieved chunks + question to GPT-4o-mini
  └── LLM generates a grounded answer
      ↓
[ Node 4: Structured Output ]
  └── Pydantic AI validates and structures the response
  └── Source attribution — which chunks answered the question
      ↓
[ LangSmith ]
  └── Traces every node with latency, tokens, and cost
```

---

## Chunking Strategies

| Strategy  | How it works                                              | Best for                        |
|-----------|-----------------------------------------------------------|---------------------------------|
| fixed     | Splits into equal 500-character chunks with 50 overlap   | Simple, fast, short documents   |
| recursive | Splits on paragraphs → sentences → words in priority     | Most documents, preserves structure |
| semantic  | Uses embeddings to find natural topic boundaries         | Long, complex financial reports |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/financial-doc-agent.git
cd financial-doc-agent
```

### 2. Configure environment variables

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=sk-your-openai-key-here
LANGCHAIN_API_KEY=your-langsmith-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=financial-doc-agent
```

Get your LangSmith API key for free at **smith.langchain.com** → Settings → API Keys.

### 3. Run with Docker
```bash
docker compose up --build
```

### 4. Or run locally
```bash
pip install -r requirements.txt
cd app
uvicorn app:app --reload
```

### 5. Access the API
- **Swagger UI** → `http://localhost:8000/docs`
- **LangSmith traces** → `https://smith.langchain.com`

---

## API Endpoints

| Method | Endpoint   | Description                                      |
|--------|------------|--------------------------------------------------|
| POST   | /analyze   | Upload a PDF and ask a question                  |
| POST   | /evaluate  | Run full evaluation suite against LangSmith dataset |
| GET    | /health    | Health check                                     |

### POST /analyze

Upload a PDF with a question and chunking strategy:

```
file: Q3_report.pdf
question: What was the net revenue growth compared to Q2?
chunking_strategy: recursive
```

Response:
```json
{
  "question": "What was the net revenue growth compared to Q2?",
  "answer": "Net revenue grew 13.5% compared to Q2.",
  "confidence": 0.9687,
  "chunking_strategy": "recursive",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "sources": [
    {
      "content": "Net revenue for Q3 2024 was $4.2M, compared to $3.7M in Q2...",
      "page": 0,
      "chunk_index": 0
    }
  ],
  "reasoning_steps": [
    "Ingested document into 2 chunks using recursive strategy",
    "Retrieved 2 relevant chunks from ChromaDB",
    "LLM generated answer from retrieved context",
    "Structured output generated with source attribution"
  ]
}
```

### POST /evaluate

Runs faithfulness, relevance, and correctness evaluators against the LangSmith dataset and logs results.

---

## LangSmith Evaluation

| Evaluator     | What it measures                                          |
|---------------|-----------------------------------------------------------|
| Faithfulness  | Is the answer grounded in the document context?           |
| Relevance     | Did retrieval return chunks relevant to the question?     |
| Correctness   | Is the answer factually correct vs. reference answer?     |

All evaluation runs are logged to LangSmith with per-run latency, token usage, and cost.

---

## Docker Commands

| Command                     | Description                                      |
|-----------------------------|--------------------------------------------------|
| `docker compose up --build` | Build image and start container                  |
| `docker compose up`         | Start without rebuilding                         |
| `docker compose stop`       | Pause container                                  |
| `docker compose down`       | Stop and remove container                        |
| `docker compose down -v`    | Stop, remove container and delete ChromaDB volume|
| `docker compose logs api`   | View API logs                                    |

---

## Generate a Test PDF

```bash
pip install fpdf2
python create_test_pdf.py
```

This creates `docs/test_report.pdf` with sample financial data you can immediately use with `/analyze`.

---

## Requirements

```
fastapi
uvicorn
langgraph
langchain
langchain-openai
langchain-community
langchain-experimental
langchain-text-splitters
langchain-huggingface
langsmith
pydantic-ai
chromadb
sentence-transformers
pypdf
python-dotenv
fpdf2
```

---

## Notes

- The `.env` file should never be committed — add it to `.gitignore`
- ChromaDB data is persisted in a named Docker volume — safe across container restarts
- HuggingFace model is downloaded on first run and cached locally
- Use `recursive` chunking as default — it performs best on most financial documents
- LangSmith traces are retained for 14 days on the free tier
