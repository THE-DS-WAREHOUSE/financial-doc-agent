import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langsmith import traceable
from chunking import get_chunks
from retrieval import store_chunks, retrieve_chunks
from schemas import AnalyzeResponse, ChunkSource
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # 0 = deterministic, best for factual financial questions
    api_key=os.getenv("OPENAI_API_KEY")
)

# --- Agent State ---
# TypedDict defines the shared state that flows between all LangGraph nodes
# Every node reads from and writes to this state
class AgentState(TypedDict):
    text: str                    # raw document text
    question: str                # user question
    chunking_strategy: str       # which chunking strategy to use
    chunks: list                 # document chunks after splitting
    retrieved: list              # chunks retrieved from ChromaDB
    answer: str                  # final answer from LLM
    confidence: float            # confidence score
    reasoning_steps: List[str]   # log of steps taken
    sources: list                # source chunks used to answer

# --- Node 1: Ingestion ---
# Splits the document into chunks using the selected strategy
# and stores them in ChromaDB
def ingest_node(state: AgentState) -> AgentState:
    chunks = get_chunks(state["text"], state["chunking_strategy"])
    store_chunks(chunks, collection_name="financial_docs")
    state["chunks"] = chunks
    state["reasoning_steps"].append(f"Ingested document into {len(chunks)} chunks using {state['chunking_strategy']} strategy")
    return state

# --- Node 2: Retrieval ---
# Queries ChromaDB for the most relevant chunks to the user question
def retrieval_node(state: AgentState) -> AgentState:
    results = retrieve_chunks(state["question"], collection_name="financial_docs")
    state["retrieved"] = results
    state["reasoning_steps"].append(f"Retrieved {len(results)} relevant chunks from ChromaDB")
    return state

# --- Node 3: Reasoning ---
# Sends the retrieved chunks + question to the LLM and gets an answer
def reasoning_node(state: AgentState) -> AgentState:
    # Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc, score in state["retrieved"]])

    prompt = f"""You are a financial analyst assistant.
Use only the context below to answer the question.
If you cannot find the answer in the context, say "I don't have enough information."

Context:
{context}

Question: {state["question"]}

Provide a clear, concise answer."""

    response = llm.invoke(prompt)
    state["answer"] = response.content
    state["reasoning_steps"].append("LLM generated answer from retrieved context")
    return state

# --- Node 4: Structured Output ---
# Builds the final structured response with source attribution
def output_node(state: AgentState) -> AgentState:
    sources = []
    for i, (doc, score) in enumerate(state["retrieved"]):
        sources.append(ChunkSource(
            content=doc.page_content[:200],  # first 200 chars of the chunk
            page=doc.metadata.get("page", 0),
            chunk_index=i
        ))

    # Confidence is derived from the similarity score of the best chunk
    # ChromaDB returns distance — lower is better, so we invert it
    best_score = state["retrieved"][0][1] if state["retrieved"] else 1.0
    confidence = round(max(0.0, 1.0 - best_score), 4)

    state["sources"] = sources
    state["confidence"] = confidence
    state["reasoning_steps"].append("Structured output generated with source attribution")
    return state

# --- Build the LangGraph ---
# Defines the flow: ingest → retrieve → reason → output → END
@traceable(name="financial-doc-agent")  # LangSmith traces this entire function
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("output", output_node)

    # Define the edges — order of execution
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "output")
    graph.add_edge("output", END)

    return graph.compile()

agent = build_agent()

# --- Run the agent ---
@traceable(name="run-financial-agent")  # LangSmith traces every individual run
def run_agent(text: str, question: str, chunking_strategy: str) -> AnalyzeResponse:
    initial_state = AgentState(
        text=text,
        question=question,
        chunking_strategy=chunking_strategy,
        chunks=[],
        retrieved=[],
        answer="",
        confidence=0.0,
        reasoning_steps=[],
        sources=[]
    )

    final_state = agent.invoke(initial_state)

    return AnalyzeResponse(
        question=question,
        answer=final_state["answer"],
        confidence=final_state["confidence"],
        chunking_strategy=chunking_strategy,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        sources=final_state["sources"],
        reasoning_steps=final_state["reasoning_steps"]
    )