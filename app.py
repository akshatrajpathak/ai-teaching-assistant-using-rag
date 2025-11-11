import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ==========================================================
# ✅ Environment Setup
# ==========================================================
load_dotenv()  # For local testing

# On Streamlit Cloud, keys come from Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ Groq API key not found. Please add it in Streamlit → Settings → Secrets.")
    st.stop()

# ==========================================================
# ✅ Directory Setup
# ==========================================================
DB_FAISS_PATH = "vectorstore/db_faiss"
UPLOADS_DIR = "uploaded_pdfs"
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ==========================================================
# ✅ Helper Functions
# ==========================================================
@st.cache_resource(show_spinner=False)
def create_vectorstore_from_pdf(uploaded_file):
    """Create FAISS vectorstore from uploaded PDF."""
    pdf_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        st.warning("⚠️ No text found in the PDF.")
        return None

    st.info("🔍 Creating embeddings... please wait.")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    db = FAISS.from_texts(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db


@st.cache_resource(show_spinner=False)
def load_existing_vectorstore():
    """Load existing FAISS index if available."""
    if os.path.exists(DB_FAISS_PATH):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception:
            st.warning("⚠️ Could not load existing FAISS index. Please upload a new PDF.")
            return None
    return None


def set_custom_prompt(custom_prompt_template):
    """Create a custom prompt template."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


# ==========================================================
# ✅ Streamlit UI
# ==========================================================
def main():
    st.set_page_config(page_title="AI Teaching Assistant", page_icon="🤖", layout="wide")

    # --- Custom Styling ---
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
            color: white;
        }
        .title-container {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #8ab4f8;
        }
        .subtitle {
            color: #b3b3b3;
            font-size: 1rem;
            margin-top: -5px;
        }
        .stChatInput {
            border-radius: 10px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Title Section ---
    st.markdown("""
        <div class='title-container'>
            <h1 class='app-title'>🤖 AI Teaching Assistant (RAG)</h1>
            <p class='subtitle'>Ask questions from your uploaded PDF using Groq LLM + FAISS Retrieval</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for PDF upload
    st.sidebar.header("📂 Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf"])

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_existing_vectorstore()

    if uploaded_file is not None:
        st.session_state.vectorstore = create_vectorstore_from_pdf(uploaded_file)
        if st.session_state.vectorstore:
            st.sidebar.success("✅ PDF processed and embeddings created!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input area
    user_input = st.chat_input("Ask something about the uploaded PDF...")

    if user_input:
        query = user_input.strip()

        # Display user's question
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Ensure FAISS database exists
        db = st.session_state.vectorstore
        if db is None:
            st.warning("Please upload a PDF first to create a knowledge base.")
            return

        try:
            # Initialize Groq LLM
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )

            # Load retrieval chain
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

            # Generate answer
            response = rag_chain.invoke({'input': query})
            answer = response.get("answer", "⚠️ No answer generated.")

            # Display assistant response
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

        # Rerun to refresh chat
        st.rerun()


# ==========================================================
# ✅ Run the App
# ==========================================================
if __name__ == "__main__":
    main()
