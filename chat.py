import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder
from qdrant_client import models

load_dotenv()

# 1-indexed, inclusive page ranges each role is allowed to retrieve from.
# `None` means no restriction (full document access).
ROLE_PAGE_RANGES = {
    "hr": (1, 6),
    "auditor": (7, 16),
    "engineer": None,
}


def build_role_filter(role: str) -> "models.Filter | None":
    page_range = ROLE_PAGE_RANGES[role]
    if page_range is None:
        return None
    start, end = page_range
    # `page` in the stored metadata is 0-indexed, while the ranges above
    # are the 1-indexed page numbers a human would ask for.
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.page",
                range=models.Range(gte=start - 1, lte=end - 1),
            )
        ]
    )

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


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Ask the LLM for alternate phrasings of the query to widen recall,
    since a single phrasing can miss chunks worded differently."""
    prompt = f"""Generate {n} different rephrasings of the question below.
Each rephrasing should preserve the original meaning but vary the wording,
so it can surface documents phrased differently than the original.
Return exactly {n} lines, one rephrasing per line, with no numbering or extra text.

QUESTION: {query}"""

    response = chat_model.invoke([{"role": "user", "content": prompt}])
    text = response.content
    if isinstance(text, list):
        text = "".join(block.get("text", "") for block in text if isinstance(block, dict))

    variants = [line.strip() for line in text.splitlines() if line.strip()]
    return [query] + variants[:n]

# Take the user's role and enforce it as a hard filter on retrieval, so
# pages outside their access never reach the LLM in the first place.
role = input("Enter your role (hr/auditor/engineer): ").strip().lower()
if role not in ROLE_PAGE_RANGES:
    raise SystemExit(f"Unknown role '{role}'. Must be one of: {', '.join(ROLE_PAGE_RANGES)}")
role_filter = build_role_filter(role)

# Take user query
user_query = input("Enter your query: ")

# Multi-query retrieval: search with the original query plus several LLM-
# generated rephrasings, then merge the results. This widens recall beyond
# what a single phrasing's embedding would match.
query_variants = generate_query_variants(user_query)

candidates_by_content = {}
for variant in query_variants:
    for doc in vector_db.similarity_search(query=variant, k=20, filter=role_filter):
        candidates_by_content[doc.page_content] = doc
candidates = list(candidates_by_content.values())

if not candidates:
    print("AI Response: No accessible content matched your query for your role.")
    raise SystemExit

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
