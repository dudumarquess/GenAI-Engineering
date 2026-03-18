from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question to be answered by the LLM.")
    
class AskResponse(BaseModel):
    answer: str