import os
import streamlit as st
from pypdf import PdfReader
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
load_dotenv()  # for local testing

# On Streamlit Cloud, keys come from Secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ Groq API key not found. Please add it in Streamlit → Settings → Secrets.")
    st.stop()

# ==========================================================
# ✅ Directory Setup
# ==========================================================
DB_FAISS_PATH = "vectorstore/db_faiss"
UPLOADS_DIR = "uploaded_pdfs"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

# ==========================================================
# ✅ Helper Functions
# ==========================================================

@st.cache_resource(show_spinner=False)
def create_vectorstore_from_pdf(uploaded_file):
    """Create FAISS vectorstore from uploaded PDF."""
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            st.warning("⚠️ No text found in the PDF.")
            return None

        st.info("📚 Creating embeddings... please wait.")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        db = FAISS.from_texts(chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        return db
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None


@st.cache_resource(show_spinner=False)
def load_existing_vectorstore():
    """Load existing FAISS index if available."""
    if os.path.exists(DB_FAISS_PATH):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"⚠️ Could not load existing FAISS index: {str(e)}")
            return None
    return None


# ==========================================================
# ✅ Streamlit UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="AI Teaching Assistant", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("📘 AI Teaching Assistant (RAG-based)")
    st.caption("Ask questions from your uploaded PDF using Groq LLM + FAISS Retrieval.")

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDF 📄")
        uploaded_file = st.file_uploader("Upload a file", type=["pdf"])
        
        st.divider()
        st.markdown("### About")
        st.info(
            "This app uses RAG (Retrieval Augmented Generation) to answer questions from your PDF documents. "
            "Upload a PDF and ask questions about its content!"
        )
        
        st.markdown("### How to use")
        st.markdown(
            """
            1. Upload a PDF file
            2. Wait for processing
            3. Ask questions about the content
            4. Get AI-powered answers!
            """
        )

    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_existing_vectorstore()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Process uploaded file
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = create_vectorstore_from_pdf(uploaded_file)
        if st.session_state.vectorstore:
            st.sidebar.success("✅ PDF processed and embeddings created!")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something about the uploaded PDF..."):
        # Ensure FAISS database exists
        if st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload a PDF first to create a knowledge base.")
            st.stop()

        # Display user's question
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Initialize Groq LLM
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )

            # Load standard retrieval QA prompt
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            rag_chain = create_retrieval_chain(
                st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}), 
                combine_docs_chain
            )

            # Get response
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({'input': prompt})
            
            answer = response.get("answer", "⚠️ No answer generated.")

            # Display bot response
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")


# ==========================================================
# ✅ Run the App
# ==========================================================
if __name__ == "__main__":
    main()
