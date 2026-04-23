import os
from dotenv import load_dotenv
from .loader import DocumentLoader
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, docs_dir="docs", store_path="faiss_index"):
        self.loader = DocumentLoader(docs_dir)
        self.embedder = Embedder()
        self.vector_store = VectorStore(self.embedder.get_embeddings(), store_path)
        self.retriever = Retriever(self.vector_store)
        
        # Determine model type from environment
        model_type = "openai" if os.getenv("OPENAI_API_KEY") else "groq"
        self.generator = Generator(model_type=model_type)

    def initialize(self, force_rebuild=False):
        """Initializes the vector store. Loads from disk or creates new if needed."""
        if force_rebuild or not os.path.exists(self.vector_store.store_path):
            print("Building vector store...")
            docs = self.loader.load_documents()
            if docs:
                self.vector_store.create_store(docs)
            else:
                print("No documents found in docs directory.")
        else:
            print("Loading existing vector store...")
            self.vector_store.load_store()

    def answer_query(self, query: str) -> str:
        """Complete RAG pipeline: Retrieve -> Generate."""
        context_docs = self.retriever.retrieve(query)
        if not context_docs:
            # Fallback to pure generation if no context found
            return self.generator.generate_answer(query, [])
            
        return self.generator.generate_answer(query, context_docs)

# Singleton instance or helper function
_rag_system = None

def get_rag_system():
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
        _rag_system.initialize()
    return _rag_system

def answer_query_with_rag(query: str) -> str:
    """Entry point function as requested."""
    rag = get_rag_system()
    return rag.answer_query(query)
