from fastapi import APIRouter, HTTPException
from app.schemas.classify_schema import ClassifyRequest, ClassifyResponse
from app.services.classify_service import classify_text

router = APIRouter()

@router.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest):
    try:
        result = classify_text(payload.text)
        return ClassifyResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
