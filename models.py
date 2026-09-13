from typing import List, Optional
from pydantic import BaseModel, Field

class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection: Optional[str] = None   # optional override

class Source(BaseModel):
    text: str = ""
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None

class Answer(BaseModel):
    answer: str
    sources: List[Source] = []

class HealthResponse(BaseModel):
    status: str
    weaviate_ready: bool