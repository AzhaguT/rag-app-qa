"""
main_ui.py
----------
Streamlit UI for a RAG (Retrieval-Augmented Generation) based Document Q&A app.
Uses TinyLlama via Ollama, Chroma vector store, and HuggingFace embeddings.
"""

import os
import streamlit as st
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

from services.pdf_parser import PDFParser
from services.embedding import EmbeddingService
from services.vector_store import VectorStore

# Disable Streamlit file watcher to avoid torch-related issues
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Set Ollama host for LLM inference (can be overridden via environment variable)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST

def get_ollama_llm(model_name="tinyllama", temperature=0.3):
    """Initialize Ollama LLM with the specified model."""
    return Ollama(model=model_name, temperature=temperature)

def main():
    # UI Setup
    st.set_page_config(page_title="RAG Document QA", layout="centered")
    st.title("📄 RAG-based Document Q&A App (Powered by TinyLlama via Ollama)")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        MAX_MB = 5  # File size limit in MB
        if uploaded_file.size > MAX_MB * 1024 * 1024:
            st.error(f"❌ File too large. Max size is {MAX_MB} MB.")
            return

        try:
            with st.spinner("🔄 Processing PDF..."):
                # Save PDF temporarily
                with open("temp_uploaded.pdf", "wb") as f:
                    f.write(uploaded_file.read())

                st.success("✅ PDF uploaded successfully!")

                # Parse and chunk PDF
                parser = PDFParser("temp_uploaded.pdf")
                full_text = parser.parse()

                if not full_text:
                    st.error("⚠️ No text found in PDF.")
                    return

                chunks = parser.chunk_text(full_text)
                if not chunks:
                    st.error("⚠️ Failed to chunk text.")
                    return

                # Wrap chunks as Documents
                documents = [Document(page_content=chunk) for chunk in chunks]

                # Generate embeddings and store in vector DB
                embedder = EmbeddingService()
                embedder.generate_embeddings(chunks)  # Optional line
                vector_store = VectorStore(persist_directory="chroma_db")
                vector_store.add_documents(documents)

            st.success("✅ Embeddings stored successfully!")

            # Set up LLM and retrieval chain
            llm = get_ollama_llm()
            retriever = vector_store.db.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            # Handle question input
            query = st.text_input("💬 Ask a question about the document:")
            if query:
                response = qa_chain(query)
                st.subheader("🔍 Answer:")
                st.write(response["result"])

                # Show source documents
                with st.expander("📚 Source(s) used"):
                    for idx, doc in enumerate(response["source_documents"], 1):
                        st.markdown(f"**Source {idx}:**")
                        st.write(doc.page_content.strip()[:500] + "...")

        except Exception as e:
            st.error(f"🔥 App error: {e}")

if __name__ == "__main__":
    main()
