import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- Streamlit Page Config ----------
st.set_page_config(
    page_title="SmartDocs Agent – KnowledgeBase AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 SmartDocs Agent – Company KnowledgeBase AI")
st.markdown("""
This AI agent answers questions from company documents (HR policies, SOPs, onboarding guides etc.)

1️⃣ Upload documents  
2️⃣ Ask a question  
""")

temperature = st.sidebar.slider("Answer creativity (temperature)", 0.0, 1.0, 0.2)
top_k = st.sidebar.slider("Search depth (chunks)", 2, 10, 4)

if not GROQ_API_KEY:
    st.error("⚠️ No GROQ_API_KEY found in .env file")
    st.stop()

# Initialize Model & Embeddings
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=temperature
)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------- Document Upload ----------
uploaded_files = st.file_uploader(
    "📁 Upload PDF/TXT/DOCX company documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

os.makedirs("temp_uploads", exist_ok=True)


@st.cache_resource(show_spinner=True)
def build_vector_store(files):
    docs = []

    for f in files:
        filename = f.name
        ext = filename.split(".")[-1].lower()
        temp_path = os.path.join("temp_uploads", filename)

        with open(temp_path, "wb") as tmp:
            tmp.write(f.read())

        if ext == "pdf":
            loader = PyPDFLoader(temp_path)
        elif ext == "txt":
            loader = TextLoader(temp_path, encoding="utf-8")
        elif ext == "docx":
            loader = Docx2txtLoader(temp_path)
        else:
            continue

        for d in loader.load():
            d.metadata["source"] = filename
            docs.append(d)

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


vector_store = None

if uploaded_files:
    st.info("🧠 Processing and indexing your documents...")
    vector_store = build_vector_store(uploaded_files)
    if vector_store:
        st.success("📚 Knowledge Base Ready!")
    else:
        st.error("⚠️ Could not extract content from files")

# ---------- Q&A ----------
st.subheader("💬 Ask a Question")

query = st.text_input("What do you want to know?")

if st.button("Get Answer") and query:
    if not vector_store:
        st.error("⚠️ Please upload documents first.")
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

        with st.spinner("🤖 Searching and analyzing..."):
            # LangChain 1.x: use invoke() instead of get_relevant_documents()
            sources = retriever.invoke(query)

            if not sources:
                st.warning("No relevant info found in uploaded docs!")
                answer = "Not enough information found in the documents provided."
            else:
                context = "\n\n".join(
                    f"Source:{d.metadata['source']}\nContent:{d.page_content}"
                    for d in sources
                )

                prompt = (
                    "You are an HR & Operations Q&A assistant for companies.\n"
                    "Answer only using the document information provided.\n"
                    "If not present, say you don't know.\n\n"
                    f"Documents:\n{context}\n\n"
                    f"Question: {query}\n"
                    "Give a clear answer in bullet points."
                )

                response = llm.invoke(prompt)
                answer = response.content

        st.markdown("### 🧠 Answer")
        st.write(answer)

        st.markdown("---")
        st.markdown("### 📄 Sources")
        for i, src in enumerate(sources, 1):
            st.markdown(f"**{i}. {src.metadata['source']}**")
            st.caption(src.page_content[:200] + "...")
