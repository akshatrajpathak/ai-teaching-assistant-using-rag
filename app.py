import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
UPLOADS_DIR = "uploaded_pdfs"
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ✅ Cache the vectorstore for uploaded PDF
@st.cache_resource(show_spinner=False)
def create_vectorstore_from_pdf(uploaded_file):
    pdf_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    db = FAISS.from_texts(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db


@st.cache_resource(show_spinner=False)
def load_existing_vectorstore():
    if os.path.exists(DB_FAISS_PATH):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    st.title("Ask Chatbot!")

    # Sidebar for PDF upload
    st.sidebar.header("Upload PDF 📄")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf"])

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_existing_vectorstore()

    if uploaded_file is not None:
        st.session_state.vectorstore = create_vectorstore_from_pdf(uploaded_file)
        st.sidebar.success("✅ PDF processed and embeddings created!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ✅ Chat input at bottom (no send button)
    user_input = st.chat_input("Ask something about the PDF...")

    # Handle user input and prevent repeat responses
    if user_input:
        user_prompt = user_input.strip()

        # Display user message immediately
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            db = st.session_state.vectorstore
            if db is None:
                st.warning("Please upload a PDF first.")
                return

            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )

            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

            response = rag_chain.invoke({'input': user_prompt})
            answer = response["answer"]

            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {str(e)}")

        # ✅ Clear input and rerun to refresh chat cleanly
        st.rerun()



if __name__ == "__main__":
    main()
