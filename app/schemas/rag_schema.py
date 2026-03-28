from pydantic import BaseModel, Field

class RagRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question to be answered by the LLM.")
    
class RagResponse(BaseModel):
    answer: str
    sources: list