import json

from app.core.model_factory import get_llm
from app.core.logger import logger
from langchain_core.messages import SystemMessage, HumanMessage


llm = get_llm()

VALID_LABELS = {"question", "complaint", "request", "other"}

def safe_parse_json(raw_response: str) -> dict | None:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        return None
    
def validate_output(parsed: dict) -> bool:
    if not isinstance(parsed, dict):
        logger.warning("Parsed output is not a dictionary")
        return False

    if "label" not in parsed or "reason" not in parsed:
        logger.warning("Parsed output missing required keys: label and/or reason")
        return False

    if not isinstance(parsed["label"], str):
        logger.warning("Parsed output contains a non-string label")
        return False

    if not isinstance(parsed["reason"], str):
        logger.warning("Parsed output contains a non-string reason")
        return False

    if parsed["label"] not in VALID_LABELS:
        logger.warning(f"Invalid label: {parsed['label']}")
        return False

    if not parsed["reason"].strip():
        logger.warning("Parsed output contains an empty reason")
        return False

    return True

def fallback_response():
    return {
        "label": "other",
        "reason": "Fallback due to invalid LLM output"
    }

def classify_text(text: str) -> dict:
    logger.info(f"Classifying text: {text}")
    system_prompt = """
        Your task is to classify the following text into one of the following categories:
        - question
        - complaint
        - request
        - other

        Return ONLY valid JSON with this exact structure:
        {
        "label": "question",
        "reason": "short explanation"
        }

        The value of "label" must be exactly one of:
        question, complaint, request, other

        Examples:
        “Can you tell me when the office opens?” → question

        “Your support team never replied.” → complaint

        “Please send me the invoice.” → request

        “The sky is very clear today.” → other

        do not include any additional text or formatting.
        do not include markdown formatting in the response.
        do not include any explanations or disclaimers outside of the JSON response.
        the label must be one of the specified categories and the reason should be concise and directly related to the content of the text.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text),
    ]
    
    try:
        response = llm.invoke(messages)
        logger.info(f"LLM response: {response.content}")
        raw_response = response.content.strip()
        
        parsed = safe_parse_json(raw_response)
        if isinstance(parsed, dict):
            label_value = parsed.get("label")
            if isinstance(label_value, str):
                parsed["label"] = label_value.lower()
        
        if not validate_output(parsed):
            logger.warning(f"invalid output from LLM: {parsed}")
            return fallback_response()
         
        return parsed
    except Exception as e:
        logger.error(f"Error processing LLM response: {str(e)}")
        return fallback_response()
    
