from fastapi import APIRouter, HTTPException
from app.schemas.llm_schema import AskRequest, AskResponse
from app.services.rag_service import rag_answer

router = APIRouter()

@router.post("/rag", response_model=AskResponse)
def rag(payload: AskRequest):
    try:
        result = rag_answer(payload.question)
        return AskResponse(answer=result["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))