import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List

class VectorStore:
    def __init__(self, embeddings, store_path: str = "faiss_index"):
        self.embeddings = embeddings
        self.store_path = store_path
        self.vector_db = None

    def create_store(self, documents: List[Document]):
        """Creates a FAISS vector store from documents."""
        if not documents:
            raise ValueError("No documents provided to create vector store.")
        
        self.vector_db = FAISS.from_documents(documents, self.embeddings)
        self.vector_db.save_local(self.store_path)
        return self.vector_db

    def load_store(self):
        """Loads an existing FAISS vector store."""
        if os.path.exists(self.store_path):
            self.vector_db = FAISS.load_local(self.store_path, self.embeddings, allow_dangerous_deserialization=True)
            return self.vector_db
        return None

    def get_retriever(self, search_kwargs={"k": 3}):
        """Returns a retriever object from the vector store."""
        if not self.vector_db:
            self.load_store()
        
        if not self.vector_db:
            return None
            
        return self.vector_db.as_retriever(search_kwargs=search_kwargs)
