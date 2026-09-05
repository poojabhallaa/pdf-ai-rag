import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning-rag",
    embedding=embedding_model,
)

# Local cross-encoder used to rerank the vector search candidates
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Take user query
user_query = input("Enter your query: ")

# Retrieve a larger candidate set from the vector store
candidates = vector_db.similarity_search(query=user_query, k=20)

# Rerank candidates by query-chunk relevance and keep the top 5
pairs = [(user_query, doc.page_content) for doc in candidates]
scores = reranker.predict(pairs)
search_results = [
    doc for doc, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
]

context = "\n\n\n".join(
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata['page_label']}\n"
    f"File Location: {result.metadata['source']}"
    for result in search_results
)

SYSTEM_PROMPT = f"""You are a helpful AI assistant, who answers
user queries based on the context provided, mention the
page number where the information is found.
If the answer is not present in the context, politely respond that you don't know.
CONTEXT: {context}
"""

response = chat_model.invoke(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
)
answer = response.content
if isinstance(answer, list):
    answer = "".join(block.get("text", "") for block in answer if isinstance(block, dict))
print("AI Response:", answer)
