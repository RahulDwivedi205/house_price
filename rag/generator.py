import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class Generator:
    def __init__(self, model_type: str = "groq", model_name: str = None):
        """Initializes the LLM generator (Groq or OpenAI)."""
        if model_type == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            self.llm = ChatGroq(
                groq_api_key=api_key, 
                model_name=model_name or "llama-3.1-8b-instant"
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            self.llm = ChatOpenAI(
                openai_api_key=api_key, 
                model_name=model_name or "gpt-4o-mini"
            )

    def generate_answer(self, query: str, context_docs: list) -> str:
        """Generates an answer based on the provided query and context."""
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = ChatPromptTemplate.from_template("""
        You are an AI Real Estate Advisor. Use the following context to answer the user's question.
        If the answer is not in the context, use your general knowledge but mention that it's general advice.
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:""")
        
        chain = create_stuff_documents_chain(self.llm, prompt)
        response = chain.invoke({"input": query, "context": context_docs})
        
        return response
