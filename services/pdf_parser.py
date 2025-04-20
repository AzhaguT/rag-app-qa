"""
pdf_parser.py
-------------
Parses and chunks PDF content using PyMuPDF.
"""

import fitz  # PyMuPDF

class PDFParser:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def parse(self) -> str:
        """Extracts text from a PDF."""
        text = ""
        try:
            with fitz.open(self.file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            raise RuntimeError(f"PDF parsing failed: {e}")

    def chunk_text(self, text: str, chunk_size=500, overlap=50) -> list:
        """Splits text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks
