from app.core.model_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


llm = get_llm(temperature=0.1)

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