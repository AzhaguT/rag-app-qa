"""
embedding.py
------------
Wraps HuggingFace Embeddings used for vectorization.
"""

from langchain.embeddings import HuggingFaceEmbeddings

class EmbeddingService:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        # Load sentence-transformers model
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def generate_embeddings(self, texts: list[str]):
        """Generate embeddings for input texts."""
        return self.model.embed_documents(texts)
