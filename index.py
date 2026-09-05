import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
pdf_path = Path(__file__).parent / "ml-book.pdf"

# Load this file in python program
loader = PyPDFLoader(file_path = pdf_path)
docs =  loader.load()

# Split the document into chunks (kept small so batches stay under the
# free-tier per-minute token limit for the embedding model)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents=docs)

# 2. Initialize the Gemini Vector Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning-rag",
    batch_size=25,
)

print("Vector store created successfully!")