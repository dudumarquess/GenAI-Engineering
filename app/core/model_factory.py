from langchain_groq import ChatGroq
from app.core.config import GROQ_API_KEY

def get_llm(temperature: float = 0.2):
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=GROQ_API_KEY)