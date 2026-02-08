from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
)

model = ChatOpenAI(model="gpt-4o")

chat_history = []


def ask_question(question):
    print(f"\nUser Query: {question}")

    # Make the question clear using conversation history
    if chat_history:
        messages = (
            [
                SystemMessage(
                    content="Give the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question without any explanation."
                ),
            ]
            + chat_history
            + [HumanMessage(content=f"New question: {question}")]
        )

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = question

    # Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents.")
    for i, doc in enumerate(docs, 1):  # (docs, 1) to start enumeration from 1 be
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"Document {i} preview:\n{preview}\n")

    combines_input = f"""Based on the following documents, please answer this question: {question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear and helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # Generate answer
    messages = (
        [
            SystemMessage(
                content="You are a helpful assistant that provides answers based on the provided documents and conversation history."
            )
        ]
        + chat_history
        + [HumanMessage(content=combines_input)]
    )

    result = model.invoke(messages)
    answer = result.content.strip()

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer:\n{answer}")
    return answer


def start_chat():
    print("Ask me questions! Type quit to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
