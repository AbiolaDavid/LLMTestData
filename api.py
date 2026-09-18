=import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from weaviate_client import (
    get_agent,
    search_collection,
    close_client,
    COLLECTION_NAME,
)
from formatter import format_answer, format_search_results, format_source

# Initialize FastAPI
app = FastAPI(
    title="SOC 101 RAG API",
    description="Query the TestingData Weaviate collection (Introduction to Sociology).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    pretty: bool = Field(True, description="Return a human-readable formatted answer")


class SourceItem(BaseModel):
    text: str | None = None
    source: str | None = None
    module_title: str | None = None
    heading: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    score: float | None = None
    citation: str | None = None


class AskResponse(BaseModel):
    answer: str
    answer_pretty: str | None = None
    sources: list[SourceItem]


class SearchResponse(BaseModel):
    query: str
    collection: str
    results: list[SourceItem]
    results_pretty: str | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok", "collection": COLLECTION_NAME}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """RAG endpoint: QueryAgent searches TestingData and generates an answer."""
    try:
        agent = get_agent()
        result = agent.ask(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}") from e

    sources: list[SourceItem] = []
    for s in getattr(result, "sources", []) or []:
        props = getattr(s, "properties", None) or {}
        item = SourceItem(
            text=props.get("text"),
            source=props.get("source"),
            module_title=props.get("module_title"),
            heading=props.get("heading"),
            page=props.get("page"),
            chunk_id=props.get("chunk_id"),
        )
        item.citation = format_source(item.model_dump())
        sources.append(item)

    raw_answer = getattr(result, "final_answer", None) or str(result)

    if req.pretty:
        pretty = format_answer(raw_answer, sources, include_sources=True)
        return AskResponse(
            answer=pretty,
            answer_pretty=pretty,
            sources=sources,
        )

    return AskResponse(answer=raw_answer, sources=sources)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(8, ge=1, le=30, description="Max number of chunks to return"),
    alpha: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Hybrid search balance (0=BM25 only, 1=vector only)",
    ),
    pretty: bool = Query(True, description="Also return a human-readable block"),
):
    """Direct hybrid search against the TestingData collection."""
    try:
        hits = search_collection(q, limit=limit, alpha=alpha)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}") from e

    results = []
    for h in hits:
        item = SourceItem(
            text=h.get("text"),
            source=h.get("source"),
            module_title=h.get("module_title"),
            heading=h.get("heading"),
            page=h.get("page"),
            chunk_id=h.get("chunk_id"),
            score=h.get("score"),
        )
        item.citation = format_source(item.model_dump())
        results.append(item)

    pretty_block = format_search_results(hits) if pretty else None

    return SearchResponse(
        query=q,
        collection=COLLECTION_NAME,
        results=results,
        results_pretty=pretty_block,
    )


@app.on_event("shutdown")
def _shutdown():
    close_client()
