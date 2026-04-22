import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from embeddings import embedding_model
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB client with persistent storage
# Data is saved to disk so it survives restarts
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def store_chunks(chunks: list, collection_name: str) -> Chroma:
    # Takes a list of LangChain document chunks and stores them in ChromaDB
    # Each chunk is embedded using HuggingFace and indexed for similarity search
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=collection_name
    )
    return vectorstore

def retrieve_chunks(query: str, collection_name: str, k: int = 3) -> list:
    # Retrieves the k most semantically similar chunks to the query
    # k=3 means we return the 3 most relevant chunks
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )
    results = vectorstore.similarity_search_with_score(query, k=k)

    # Returns list of (document, score) tuples
    # Lower score = more similar in ChromaDB's distance metric
    return results