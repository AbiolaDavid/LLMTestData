import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import Question, Answer, HealthResponse
from .query_agent import ask
from .weaviate_client import get_client, close_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm the client so the first request isn't slow
    try:
        get_client()
        log.info("Weaviate client initialized")
    except Exception as e:
        log.error(f"Failed to init Weaviate: {e}")
    yield
    # Shutdown
    close_client()
    log.info("Weaviate client closed")

app = FastAPI(title="Weaviate QueryAgent API", lifespan=lifespan)

# ---------- CORS ----------
origins_env = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Routes ----------
@app.get("/", response_model=HealthResponse)
def root():
    ready = False
    try:
        ready = get_client().is_ready()
    except Exception:
        pass
    return HealthResponse(status="ok", weaviate_ready=ready)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ask", response_model=Answer)
def ask_question(payload: Question):
    try:
        result = ask(payload.question, payload.collection)
        return Answer(**result)
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@app.exception_handler(Exception)
async def unhandled(request, exc):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal error"})v