import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()

# ==========================
# Configure Gemini
# ==========================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    "gemini-3.8-flash"
)

# ==========================
# Load Embedding Model
# ==========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Load FAISS Vector Store
# ==========================
db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FinBase RAG Assistant Ready!")

# ==========================
# Chat Loop
# ==========================
while True:

    question = input(
        "\nAsk a financial question (or type exit): "
    )

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = db.similarity_search(
        question,
        k=3
    )

    # Combine retrieved context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # ==========================
       # ==========================
    # Debug Output
    # ==========================
    print("\n===== RETRIEVED CONTEXT =====\n")
    print(context[:1500])

    print("\n===== END OF CONTEXT =====")

    print("\nQUESTION:")
    print(question)

    print("\nCONTEXT LENGTH:")
    print(len(context))

    # ==========================
    # Prompt
    # ==========================
    prompt = f"""
You are a FinTech Customer Support Assistant.

You must answer ONLY from the provided context.

If the answer exists in the context:
- Give the answer directly.
- Quote relevant values, limits, clauses, or policies.
- Be concise.
- For Indian currency, write INR instead of the ₹ symbol.

If the answer does NOT exist in the context:
Reply exactly:
I could not find this information in the knowledge base.

Context:
{context}

Question:
{question}

Answer:
"""

    # ==========================
    # Gemini Call
    # ==========================
    try:
        print("\nSending prompt to Gemini...")

        response = model.generate_content(prompt)

        print("Gemini responded!")

        print("\n===== ANSWER =====\n")
        print(response.text)

    except Exception as e:
        print("\n===== GEMINI ERROR =====")
        print(str(e))

    # ==========================
    # Sources
    # ==========================
    print("\n===== SOURCES =====")

    for i, doc in enumerate(docs, start=1):
        print(f"{i}. {doc.metadata.get('source', 'Unknown')}")
        
        
