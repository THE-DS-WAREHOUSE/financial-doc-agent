from pydantic import BaseModel
from typing import Optional, List

class AnalyzeRequest(BaseModel):
    question: str
    chunking_strategy: str = "semantic"  # fixed, semantic, recursive

class ChunkSource(BaseModel):
    content: str
    page: int
    chunk_index: int

class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    chunking_strategy: str
    embedding_model: str
    sources: List[ChunkSource]
    reasoning_steps: List[str]
    langsmith_trace_url: Optional[str] = None