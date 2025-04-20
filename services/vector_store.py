"""
vector_store.py
---------------
Handles embedding storage and similarity search using ChromaDB with DuckDB backend.
"""

import logging
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from chromadb.config import Settings
from chromadb import PersistentClient

class VectorStore:
    def __init__(self, persist_directory: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Set dynamic directory if not provided
        if not persist_directory:
            persist_directory = f"./chroma_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.persist_directory = persist_directory

        self.logger.info(f"Using vector store directory: {self.persist_directory}")

        # Load sentence-transformer model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Configure Chroma to use DuckDB as backend
        self.chroma_settings = Settings(chroma_db_impl="duckdb")

        # Create a persistent client with DuckDB
        self.client = PersistentClient(path=self.persist_directory, settings=self.chroma_settings)

        # Initialize Chroma vector store
        self.db = Chroma(
            client=self.client,
            collection_name="rag_collection",
            embedding_function=self.embedding_model
        )

    def add_documents(self, documents: list[Document]):
        """Add documents to the vector store."""
        if not all(isinstance(doc, Document) for doc in documents):
            raise TypeError("All items in 'documents' must be of type langchain.schema.Document")

        try:
            self.db.add_documents(documents)
            self.db.persist()
            self.logger.info(f"Stored {len(documents)} documents in vector DB.")
        except Exception as e:
            self.logger.error(f"Error storing vectors: {e}")
            raise

    def query(self, query_text: str, k: int = 3) -> list[Document]:
        """Search similar documents based on query text."""
        try:
            results = self.db.similarity_search(query_text, k=k)
            for res in results:
                if not isinstance(res, Document):
                    raise TypeError("Query result is not a Document object.")
            return results
        except Exception as e:
            self.logger.error(f"Error querying the vector store: {e}")
            raise
