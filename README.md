# 🧠 RAG-Based Q&A App (Streamlit + LangChain)

This is a Retrieval-Augmented Generation (RAG) app using:
- Streamlit for UI
- LangChain for LLM integration
- ChromaDB for vector store
- PyMuPDF for PDF parsing

## 🔧 How to Run Locally

```bash
git clone https://github.com/your-username/rag_app.git
cd rag_app
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
streamlit run main_ui.py
