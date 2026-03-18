from fastapi import FastAPI
from app.routes.llm_routes import router as llm_router

app = FastAPI(title="GenAI Engineering API", description="A simple API to interact with a language model.")

app.include_router(llm_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the GenAI Engineering API. Use the /ask endpoint to interact with the language model."}