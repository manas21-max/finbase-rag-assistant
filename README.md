# FinBase RAG Assistant

An AI-powered FinTech customer support assistant that answers questions using information retrieved from a financial knowledge base. The project uses Retrieval-Augmented Generation (RAG) to ground responses in relevant document content.

## Features

- Answers questions about financial documents.
- Retrieves relevant document chunks using semantic similarity search.
- Generates responses using Google's Gemini model.
- Displays retrieved source documents.
- Uses a fallback response when the answer cannot be found in the knowledge base.
- Provides a Streamlit web interface.

## Technology Stack

- **Frontend:** Streamlit
- **Language:** Python
- **LLM:** Google Gemini (`gemini-3.8-flash`, as configured in the project)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** FAISS
- **Document loading:** LangChain community PDF loader
- **Text splitting:** Recursive character text splitter

## RAG Architecture

1. **Document ingestion:** PDF documents are loaded from the `data/` directory.
2. **Preprocessing and chunking:** Document text is split into smaller chunks for retrieval.
3. **Embedding generation:** Text chunks are converted into vector embeddings using the MiniLM embedding model.
4. **Vector storage:** FAISS stores the embeddings for similarity search.
5. **Retrieval:** A user question is embedded and relevant document chunks are retrieved.
6. **Generation:** The retrieved context and user question are passed to Gemini to generate an answer.
7. **Response:** The application displays the answer and retrieved sources. If the information is not available in the knowledge base, a fallback message is returned.

## Project Structure

```text
finbase-rag-assistant/
├── app.py
├── backend/
│   ├── ingest.py
│   ├── query.py
│   └── rag_chat.py
├── data/
│   ├── sample_1.pdf
│   ├── sample_2.pdf
│   └── ...
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/manas21-max/finbase-rag-assistant.git
cd finbase-rag-assistant
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key. Do not commit `.env` or expose your API key publicly.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key used to access the Gemini model |

For Streamlit Community Cloud, configure this variable through the application's Secrets settings.

## Model and Vector Database Details

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** FAISS
- **Generation model:** Gemini, configured as `gemini-3.8-flash`
- **Retrieval method:** Semantic similarity search over stored document embeddings

## Evaluation and Testing

The application was manually tested with questions about the UPI transaction limit and the minimum balance for a FinBase Digital Savings Account. An out-of-scope question, such as asking for Google's CEO, was also tested to verify the fallback response when information is absent from the knowledge base.

These are functional checks rather than a comprehensive quantitative evaluation. Future evaluation could include a curated question-answer dataset, retrieval precision, answer faithfulness, and response latency.

## Limitations and Future Improvements

- Answer generation depends on Gemini API availability and quota.
- Retrieval quality depends on document quality, chunking, and embedding similarity.
- A larger evaluation dataset would help measure reliability.
- Future improvements include hybrid retrieval, reranking, automated evaluation, improved source citations, and clearer error handling.

## Public Links

- **GitHub repository:** https://github.com/manas21-max/finbase-rag-assistant
- **Live application:** https://finbase-rag-assistant-manas.streamlit.app/
