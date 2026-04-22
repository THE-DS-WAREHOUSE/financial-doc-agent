from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_chunks(text: str, strategy: str) -> list:

    # --- Fixed-size chunking ---
    # Splits text into equal chunks of 500 characters
    # Simple and fast but may cut sentences mid-way
    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,  # overlap avoids losing context at chunk boundaries
            separator="\n"
        )

    # --- Recursive chunking ---
    # Tries to split on paragraphs first, then sentences, then words
    # Smarter than fixed — respects document structure
    elif strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]  # priority order of split points
        )

    # --- Semantic chunking ---
    # Uses embeddings to find natural topic boundaries in the text
    # Most intelligent strategy — keeps related sentences together
    elif strategy == "semantic":
        splitter = SemanticChunker(
            embeddings=OpenAIEmbeddings(),
            breakpoint_threshold_type="percentile"  # splits where meaning changes most
        )

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Use fixed, recursive, or semantic.")

    return splitter.create_documents([text])