import logging
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

class VectorStore:
    def __init__(self, persist_directory: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Set dynamic directory if not provided
        if not persist_directory:
            persist_directory = f"./faiss_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.persist_directory = persist_directory

        self.logger.info(f"Using FAISS vector store directory: {self.persist_directory}")

        # Load sentence-transformer model for embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Initialize FAISS vector store (initially empty)
        self.db = None

    def add_documents(self, documents: list[Document]):
        """Add documents to the vector store."""
        from langchain.vectorstores import FAISS

        # Ensure all documents are of type Document
        if not all(isinstance(doc, Document) for doc in documents):
            raise TypeError("All items in 'documents' must be of type langchain.schema.Document")

        # Build and store FAISS index from the documents
        self.db = FAISS.from_documents(documents, self.embedding_model)
        self.db.save_local(self.persist_directory)
        self.logger.info(f"Stored {len(documents)} documents in FAISS vector store.")

    def query(self, query_text: str, k: int = 3) -> list[Document]:
        """Search similar documents based on query text."""
        if not self.db:
            self.db = FAISS.load_local(self.persist_directory, self.embedding_model)

        try:
            results = self.db.similarity_search(query_text, k=k)
            for res in results:
                if not isinstance(res, Document):
                    raise TypeError("Query result is not a Document object.")
            return results
        except Exception as e:
            self.logger.error(f"Error querying the vector store: {e}")
            raise
