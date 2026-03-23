from pydantic import BaseModel, Field

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to be classified by LLM")
    
class ClassifyResponse(BaseModel):
    label: str = Field(..., description="The predicted label for the input text.")
    reason: str = Field(..., description="The reason for the predicted label.")