from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Search for relevant documents
query = "Which island does SpaceX lease for its launches in the Pacific?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity-score-threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3,  # Only return chunks with cosine similarity above 0.3
#     },
# )

relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")
# Display
print("--- Context ---")
for i, doc in enumerate(relevant_docs):
    print(f"\nDocument {i+1}:")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Content length: {len(doc.page_content)} characters")
    print(f"  Content preview: {doc.page_content[:200]}...")
