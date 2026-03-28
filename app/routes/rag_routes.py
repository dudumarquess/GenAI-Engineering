from fastapi import APIRouter, HTTPException
from app.schemas.rag_schema import RagRequest, RagResponse
from app.services.rag_service import rag_answer

router = APIRouter()

@router.post("/rag", response_model=RagResponse)
def rag(payload: RagRequest):
    try:
        result = rag_answer(payload.question)
        return RagResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))