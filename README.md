# SmartDocs Agent – Company KnowledgeBase AI

SmartDocs Agent is an AI-powered **Knowledge Base Assistant** that answers questions from internal company documents such as HR policies, SOPs, onboarding guides and FAQs.

Instead of searching manually in PDFs and documents or repeatedly asking HR/Managers, employees can upload documents once and then ask questions in simple natural language.

---

## 1. Overview

- **Category:** KnowledgeBase Agent – “Answer questions from company documents”
- **Use case:** HR / Operations / Support
- **Goal:** Provide quick, accurate answers based on internal documents using Retrieval-Augmented Generation (RAG).

The agent:
1. Ingests documents (PDF, TXT, DOCX)
2. Splits text into chunks and creates vector embeddings
3. Stores them in a FAISS vector database
4. Retrieves the most relevant chunks for a user’s query
5. Uses an LLM (Groq Llama 3.1) to generate a final answer with sources

---

## 2. Features

- 📁 **Multi-format document upload** – supports PDF, TXT and DOCX
- 🔍 **Semantic search over documents** – not just keyword matching
- 🧠 **RAG-based answers** – answers are grounded in uploaded content
- 📚 **Source citations** – shows which document chunks were used
- 🎛️ **Configurable search depth & creativity** – controlled from sidebar
- 🌐 **Works locally and on Streamlit Cloud**

---

## 3. Architecture

High-level flow:

1. **User Interface (Streamlit)**
   - Document upload widget
   - Text box for questions
   - Parameters: temperature, number of chunks (k)

2. **Document Ingestion Layer**
   - Loaders:
     - `PyPDFLoader` for PDFs
     - `TextLoader` for .txt files
     - `Docx2txtLoader` for .docx files
   - Store file name in metadata (`source`)

3. **Preprocessing & Vector Store**
   - `RecursiveCharacterTextSplitter` splits documents into overlapping chunks
   - `SentenceTransformerEmbeddings` (`all-MiniLM-L6-v2`) creates dense embeddings
   - `FAISS` vector store stores all document chunks + embeddings

4. **Retrieval & Reasoning**
   - For each user query:
     - Convert query to embedding
     - Use `vector_store.as_retriever(k)` to retrieve top-k relevant chunks
   - Build a **context prompt** from retrieved chunks
   - Send the prompt to Groq LLM:
     - Model: `llama-3.1-8b-instant`
     - Via `langchain_groq.ChatGroq`

5. **Response Generation**
   - LLM returns an answer
   - Streamlit shows:
     - Final answer
     - List of source documents and snippets

You can draw this as a block diagram with the following boxes:

- **User (Browser)** → **Streamlit UI**
- **Streamlit UI** → **Document Loaders** → **Text Splitter** → **SentenceTransformer Embeddings** → **FAISS Vector Store**
- **User Query** → **Retriever (from FAISS)** → **Groq LLM (Llama 3.1)** → **Answer + Sources → Streamlit UI**

---

## 4. Tech Stack

- **Frontend / UI:** Streamlit
- **LLM Provider:** Groq Cloud
- **LLM Model:** `llama-3.1-8b-instant`
- **Framework:** LangChain (with `langchain-community`, `langchain-text-splitters`, `langchain-groq`)
- **Embeddings:** `SentenceTransformerEmbeddings` (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS (in-memory)
- **Document Loaders:**
  - `PyPDFLoader`
  - `TextLoader`
  - `Docx2txtLoader`
- **Language:** Python 3.x

---

5. Setup & Run Instructions (Local)

Prerequisites

- Python 3.10+ installed
- Groq API key from https://console.groq.com/

 Steps

1. **Clone the repo**

```bash
git clone https://github.com/YashaswiniKurabet/Smartdocs-agent.git
cd Smartdocs-agent
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit application**

```bash
streamlit run app.py
```

---

## 6. License

This project is licensed under the [MIT License](LICENSE).

