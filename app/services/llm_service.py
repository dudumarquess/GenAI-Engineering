from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import GROQ_API_KEY

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1,groq_api_key=GROQ_API_KEY)

print("Language Model initialized with Groq.")

def ask_llm(question: str) -> str:
    system_prompt = (
        "You are a helpful assistant that provides concise answers to user questions. "
        "Answer the question clearly and concisely, without unnecessary information. If you don't know the answer, say you don't know. "
        "If the question is ambiguous, ask for clarification. Always provide a direct answer to the question asked."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    
    try:
        response =llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error communicating with LLM: {str(e)}")