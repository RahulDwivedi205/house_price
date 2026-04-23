from typing import List
from langchain_core.documents import Document

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieves top k relevant chunks for a given query."""
        retriever_obj = self.vector_store.get_retriever(search_kwargs={"k": k})
        if not retriever_obj:
            return []
        
        return retriever_obj.get_relevant_documents(query)
