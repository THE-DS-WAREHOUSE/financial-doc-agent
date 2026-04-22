import os
import pypdf
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from agent import run_agent
from evaluation import create_eval_dataset, run_evaluation
from schemas import AnalyzeRequest, AnalyzeResponse
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the LangSmith evaluation dataset on startup
    # Skips if the dataset already exists
    try:
        create_eval_dataset()
        print("LangSmith evaluation dataset ready.")
    except Exception as e:
        print(f"Dataset already exists or error: {e}")
    yield

app = FastAPI(
    title="Financial Document Intelligence Agent",
    description="LangGraph agent for financial document Q&A with chunking strategies, HuggingFace embeddings, and LangSmith evaluation.",
    version="1.0.0",
    lifespan=lifespan
)

# --- POST /analyze ---
# Accepts a PDF file and a question, runs the full LangGraph agent pipeline
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile = File(...),
    question: str = "What are the key financial highlights?",
    chunking_strategy: str = "recursive"
):
    # Validate chunking strategy
    if chunking_strategy not in ["fixed", "recursive", "semantic"]:
        raise HTTPException(
            status_code=400,
            detail="chunking_strategy must be one of: fixed, recursive, semantic"
        )

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Extract text from the uploaded PDF
    try:
        pdf_reader = pypdf.PdfReader(file.file)
        text = "\n".join([
            page.extract_text()
            for page in pdf_reader.pages
            if page.extract_text()
        ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    # Run the LangGraph agent
    try:
        result = run_agent(
            text=text,
            question=question,
            chunking_strategy=chunking_strategy
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return result

# --- POST /evaluate ---
# Runs the full evaluation suite against the LangSmith dataset
# Returns faithfulness, relevance, and correctness scores
@app.post("/evaluate")
async def evaluate_agent():
    try:
        def agent_function(inputs: dict):
            result = run_agent(
                text=inputs.get("context", ""),
                question=inputs.get("question", ""),
                chunking_strategy="recursive"
            )
            return {
                "answer": result.answer,
                "context": inputs.get("context", "")
            }

        results = run_evaluation(agent_function)
        return {
            "message": "Evaluation complete. Check LangSmith for detailed results.",
            "langsmith_url": f"https://smith.langchain.com/o/{os.getenv('LANGCHAIN_PROJECT')}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

# --- GET /health ---
# Simple health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "agent": "financial-doc-agent"}