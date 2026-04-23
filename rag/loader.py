import os
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, docs_dir: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self) -> List[Document]:
        """Loads and splits documents from the docs directory."""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            return []

        # Load Text files
        txt_loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        # Load PDF files
        pdf_loader = DirectoryLoader(self.docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        
        documents = []
        try:
            documents.extend(txt_loader.load())
        except Exception as e:
            print(f"Error loading text documents: {e}")
            
        try:
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error loading PDF documents: {e}")

        if not documents:
            return []

        split_docs = self.text_splitter.split_documents(documents)
        return split_docs
