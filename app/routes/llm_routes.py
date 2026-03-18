from fastapi import APIRouter, HTTPException

from app.schemas.llm_schema import AskRequest, AskResponse
from app.services.llm_service import ask_llm

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        answer = ask_llm(payload.question)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))