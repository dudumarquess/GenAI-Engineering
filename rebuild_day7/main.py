from fastapi import FastAPI, HTTPException
from classify_service import classify_text
from classify_schema import ClassifyRequest, ClassifyResponse


app = FastAPI()

@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest):
    try:
        result = classify_text(payload.text)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
@app.get("/")
def app_root():
    return {"message": "This is the classification text app using LLMs"}