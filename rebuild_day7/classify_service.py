import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import GROQ_API_KEY
from logger import logger


llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=GROQ_API_KEY)

VALID_LABELS = {"question", "complaint", "request", "other"}

def fallback_response():
    return {
        "label": "other",
        "reason": "Fallback due to invalid model output"
    }

    
def classify_text(text: str) -> dict:
    system_prompt = """
        You are a text classification, that the goal is classify a string text into the following 4 labels:
        question,
        complaint,
        request,
        other.
        
        I want the result to be a JSON in the following format:
        {
            "label": "the label that the text correspond",
            "reason": "a concise and small reason of the text be the label"
        }
        
        - You have to return only the JSON, without any additional text.
        - If you don't know the answer return the label as "other" and the reason explaining that you dont know.
        - The label must be one of these: {question, complaint, request, other}
        
        Examples: 
        text: "what is the name of the US president"
        return:
        {
            "label": "question",
            "reason": "its making a question of who is the US president"
        }
    """
    
    logger.info(f"text to be classified: {text}")
    
    message = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text),
    ]
    
    
    
    
    try:
        response = llm.invoke(message).content.strip()
        logger.info(response)
        response_json = json.loads(response)
        
        if not isinstance(response_json, dict):
            return fallback_response()
             
        if "label" not in response_json or "reason" not in response_json:
            logger.error("the response does not contain label or reason")
            return fallback_response()
        
        if not isinstance(response_json["label"],str) or response_json["label"].lower() not in VALID_LABELS:
            logger.error("the label in the output is not a string or is not a valid label")
            return fallback_response()
        
        if not isinstance(response_json["reason"], str):
            logger.error("the reason is not a string")
            return fallback_response()
        
        response_json["label"] = response_json["label"].lower()
        return response_json
    except json.JSONDecodeError as e:
        logger.error(f"error trying to parse json: {str(e)}")
        return {
            "label": "other",
            "reason": "some error ocurred trying to parse to JSON"}
    except Exception as e:
        logger.error(f"error trying to parse json {str(e)}")
        return {
            "label": "other",
            "reason": "some error ocurred trying to classify the text"}
    
   