from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the embedding model using sentence-transformers."""
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def get_embeddings(self):
        return self.embeddings
