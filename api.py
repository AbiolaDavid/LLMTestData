import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from weaviate_client import client  # your Weaviate connection

app = FastAPI()

# ---------- CORS ----------
# Replace "*" with your frontend URL once it's deployed:
# e.g. ["https://myapp.vercel.app", "http://localhost:5173"]
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "ok", "service": "testingdata-llm"}

@app.post("/ask", response_model=Answer)
def ask_question(data: Question):
    try:
        collection = client.collections.get("TestingData")  # <-- change me

        results = collection.query.near_text(
            query=data.question,
            limit=3,
        )

        if not results.objects:
            return {"answer": "I couldn't find anything relevant."}

        # Combine the top matches into one answer.
        # Adjust based on your schema: obj.properties.get("text") or similar.
        snippets = [
            obj.properties.get("text", "source", "page", "chunk_id")
            for obj in results.objects
        ]
        answer = "\n\n".join(s for s in snippets if s)

        return {"answer": answer or "No content found."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weaviate error: {e}")