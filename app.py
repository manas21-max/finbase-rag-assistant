
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="FinBase RAG Assistant",
    page_icon="💳"
)

st.title("💳 FinBase RAG Assistant")
st.write("Ask questions about the financial documents in the knowledge base.")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Configure it in your environment or Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.8-flash")


# Load the existing FAISS knowledge base
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


# User question
question = st.text_input("Your question")

if st.button("Ask") and question.strip():

    with st.spinner("Searching the knowledge base..."):
        try:
            # Retrieve relevant document chunks
            docs = db.similarity_search(question, k=4)

            if not docs:
                st.subheader("Answer")
                st.write(
                    "I could not find this information in the knowledge base."
                )
                st.stop()

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # Build a context-grounded prompt
            prompt = f"""
You are a FinTech Customer Support Assistant.

Answer the question using ONLY the provided context.
Do not invent facts or use outside knowledge.

If the answer exists in the context:
- Answer directly and concisely.
- Preserve relevant values, limits, and conditions.
- Write Indian currency amounts using INR instead of the rupee symbol.

If the answer does not exist in the context, reply exactly:
I could not find this information in the knowledge base.

Context:
{context}

Question:
{question}

Answer:
"""

            # Generate the answer
            response = model.generate_content(prompt)

            answer = getattr(response, "text", None)

            st.subheader("Answer")

            if answer and answer.strip():
                # Avoid the square glyph when displaying rupee amounts
                answer = answer.replace("₹", "INR ")
                st.write(answer.strip())
            else:
                st.warning(
                    "The AI model returned an empty response. "
                    "Please try again."
                )

            # Show retrieved source documents
            with st.expander("Retrieved sources"):
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get(
                        "source", "Unknown source"
                    )
                    st.write(f"{i}. {source}")

        except Exception as exc:
            error_text = str(exc)

            if "429" in error_text or "quota" in error_text.lower():
                st.warning(
                    "The AI service has temporarily reached its "
                    "request quota. Please try again later."
                )
            else:
                st.error(
                    f"Unable to answer this question: {error_text}"
                )
