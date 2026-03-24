from app.core.model_factory import get_llm
from app.core.logger import logger
from langchain_core.messages import SystemMessage, HumanMessage

llm = get_llm()

def load_docs():
    source_path = "./data/docs.txt"
    with open(source_path, "r") as f:
        logger.info(f"Loaded documents from {source_path}")
        return f.read().split("\n\n")

def retrieve_chunks(query, chunks):
    return [chunk for chunk in chunks if any(word in chunk.lower() for word in query.lower().split())]


def rag_answer(question: str) -> dict:
    chunks = load_docs()
    relevant_chunks = retrieve_chunks(question, chunks)
    logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for the question: '{question}'")
    if not relevant_chunks:
        return {"answer": "No relevant information found."}
    
    
    context = "\n".join(relevant_chunks)
    
    system_prompt = """
        You are an assistant that must answer only using the provided context.
        If the answer is not contained in the context, say "I don't know."
        Be concise and do not invent information.
        """

    human_prompt = f"""
    Context:
    {context}

    Question:
    {question}
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        return {"answer": "An error occurred while processing your request."}
    
    