import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all the documents from the specified directory."""
    print(f"Loading documents from {docs_path}...")

    # Check if docs_path exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory '{docs_path}' does not exist.")

    # Load all text files from the directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        },  # Ensure UTF-8 encoding to handle special characters
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No text files found in the directory '{docs_path}'.")

    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks."""
    print(
        f"\nSplitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}..."
    )

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(
            chunks[:5]
        ):  # Print the first 5 chunks for verification
            print(f"\nChunk {i+1}:")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"  Content length: {len(chunk.page_content)} characters")
            print(f"  Content preview: {chunk.page_content}")
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n...and {len(chunks) - 5} more chunks.")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=100)

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and persisted to {persist_directory}")
    return vectorstore


def main():
    print("Main function")

    # 1. Load documents
    documents = load_documents(docs_path="docs")

    # 2. Split documents into chunks
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)

    # 3. Create and persist vector store
    vectorstore = create_vector_store(chunks, persist_directory="db/chroma_db")


if __name__ == "__main__":
    main()
