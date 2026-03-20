import json

from app.core.model_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


llm = get_llm()

def classify_text(text: str) -> dict:
    system_prompt = """
Your task is to classify the following text into one of the following categories:
- Question
- Complaint
- Request
- Other

Return only a valid output JSON:
{
    "label": "Question" | "Complaint" | "Request" | "Other",
    "reason": "A brief explanation of why the text was classified into this category."
}

Examples:
“Can you tell me when the office opens?” → question

“Your support team never replied.” → complaint

“Please send me the invoice.” → request

“The sky is very clear today.” → other

do not include any additional text or formatting.
do not include markdown formatting in the response.
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text),
    ]
    
    try:
        response = llm.invoke(messages)
        raw_response = response.content.strip()
        
        parsed = json.loads(raw_response)
        
        if "label" not in parsed or "reason" not in parsed:
            raise ValueError("Missing 'label' or 'reason' in LLM response.")
        return parsed
    except json.JSONDecodeError:
        raise ValueError(f"LLM response is not valid JSON: {raw_response}")
    except Exception as e:
        raise ValueError(f"Error processing LLM response: {str(e)}")