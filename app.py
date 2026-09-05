import os
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

load_dotenv()
QDRANT_URL = "http://localhost:6333"


@st.cache_resource
def load_models():
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedding_model, chat_model, reranker


embedding_model, chat_model, reranker = load_models()

st.title("PDF Semantic Search & Chat")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    # Same PDF content always maps to the same collection, so re-uploading
    # doesn't re-embed and re-store it every time.
    collection_name = "pdf-" + hashlib.sha256(file_bytes).hexdigest()[:16]

    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(collection_name):
        with st.spinner("Indexing PDF..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            loader = PyPDFLoader(file_path=tmp_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents=docs)

            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embedding_model,
                url=QDRANT_URL,
                collection_name=collection_name,
                batch_size=25,
            )
            os.remove(tmp_path)
        st.success(f"Indexed {uploaded_file.name} ({len(chunks)} chunks)")
    else:
        st.info(f"{uploaded_file.name} is already indexed")

    vector_db = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    user_query = st.text_input("Ask a question about the PDF")

    if user_query:
        with st.spinner("Searching..."):
            candidates = vector_db.similarity_search(query=user_query, k=20)

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

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Retrieved chunks used for this answer"):
            for result in search_results:
                st.markdown(f"**Page {result.metadata['page_label']}**")
                st.write(result.page_content)
