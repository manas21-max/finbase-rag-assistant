import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


# Load PDFs
DATA_FOLDER = "data"
documents = []

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_FOLDER, file)

        print(f"Loading {file}...")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        documents.extend(docs)

print(f"\nLoaded {len(documents)} pages")


# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# Embeddings
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Generating embeddings...")


# Create FAISS vector store
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Save vector store
vectorstore.save_local("vectorstore")

print("\nFAISS index created successfully!")
print("Saved in vectorstore folder")
