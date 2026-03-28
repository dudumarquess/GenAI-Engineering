import os
import re

from app.core.model_factory import get_llm
from app.core.logger import logger
from langchain_core.messages import SystemMessage, HumanMessage

llm = get_llm()

STOPWORDS = {"the", "is", "a", "an", "what", "how", "does", "do", "i", "my", "to", "of", "and", "in"}

def normalize_text(text):
    return re.findall(r"\b\w+\b", text.lower())

def load_docs(folder_path="data"):
    logger.info("Loading documents from %s", folder_path)
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                text = file.read()
                logger.info("Loaded document: %s", filename)
                documents.append({
                    "source": filename,
                    "text": text,
                })
    logger.info("Collected %d document(s) for RAG processing", len(documents))
    return documents

    """ The chunking algorithm is basically fixed size word chunking with overlap, so if the relevant information
    in the chunks are divided in two chunks they are going to complement the information.
    """
def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    step = max(chunk_size - overlap, 1)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += step
    logger.debug(
        "Split text into %d chunks (chunk_size=%d overlap=%d)",
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks
        
def build_chunks(documents):
    logger.info("Building chunk index from %d document(s)", len(documents))
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        logger.debug(
            "Document %s produced %d chunk(s)",
            doc["source"],
            len(chunks),
        )
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_index": idx,
            })
    logger.info("Built %d chunk(s) total", len(all_chunks))
    return all_chunks


    """ The retrieve chunking function get all the words of the query and assign scores to the keyword matching
    and sort from the most frequent, and then return the top 3 chunks with more score
    """
def retrieve_chunks(query, chunks, top_k=3):
    query_words = [w for w in normalize_text(query) if w not in STOPWORDS]

    scored = []

    for chunk in chunks:
        chunk_words = set(normalize_text(chunk["text"]))
        score = sum(word in chunk_words for word in query_words)

        if score > 0:
            scored.append({
                **chunk,
                "score": score,
            })
            logger.debug(
                "Chunk %s index %d scored %d for query '%s'",
                chunk["source"],
                chunk["chunk_index"],
                score,
                query,
            )

    if not scored:
        logger.warning("No chunks matched the query: '%s'", query)
        return []

    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = scored[:top_k]
    logger.info("Selected %d chunk(s) for query '%s'", len(selected), query)
    return selected


def rag_answer(question: str) -> dict:
    logger.info("Processing question: %s", question)
    all_docs = load_docs()
    all_chunks = build_chunks(all_docs)
    relevant_chunks = retrieve_chunks(question, all_chunks)

    if not relevant_chunks:
        logger.warning("No relevant chunks found for question: '%s'", question)
        return {"answer": "No relevant information found."}

    sources = [chunk["source"] for chunk in relevant_chunks]
    logger.info(
        "Retrieved %d relevant chunk(s) for the question: '%s'",
        len(relevant_chunks),
        question,
    )

    for chunk in relevant_chunks:
        logger.info(
            "Chunk content (source=%s idx=%d): %s",
            chunk["source"],
            chunk["chunk_index"],
            chunk["text"],
        )

    context = "\n".join(chunk["text"] for chunk in relevant_chunks)
    logger.debug(
        "Context built from %d chunk(s) totaling %d characters",
        len(relevant_chunks),
        len(context),
    )
    
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
        return {"answer": response.content, "sources": sources}
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        return {"answer": "An error occurred while processing your request.", "sources": []}

    
