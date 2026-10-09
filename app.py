
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(
    page_title="FinBase RAG Assistant",
    page_icon="💳"
)

st.title("💳 FinBase RAG Assistant")
st.write("Ask questions about the financial documents in the knowledge base.")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Configure it in your environment.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.8-flash")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

try:
    db = load_vectorstore()
except Exception as exc:
    st.error(f"Could not load the knowledge base: {exc}")
    st.stop()

question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("Searching the knowledge base..."):
        try:
            docs = db.similarity_search(question, k=4)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = f"""
Answer the question using only the context below.
If the answer is not available in the context, reply exactly:
I could not find this information in the knowledge base.

Context:
{context}

Question:
{question}
"""
            response = model.generate_content(prompt)
            st.subheader("Answer")
            st.write(response.text)

            with st.expander("Retrieved sources"):
                for i, doc in enumerate(docs, 1):
                    st.write(
                        f"{i}. {doc.metadata.get('source', 'Unknown')}"
                    )
       
        except Exception as exc:
            error_text = str(exc)

            if "429" in error_text or "quota" in error_text.lower():
                st.warning(
                    "The AI service has temporarily reached its request "
                    "quota. Please try again later."
                )
            else:
                st.error(
                    f"Unable to answer this question: {error_text}"
                )
