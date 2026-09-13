import os
from typing import Optional
from weaviate.agents.query import QueryAgent
from .weaviate_client import get_client

# Configure via env so you can change collections without redeploying code
COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "TestingData")

_agent: Optional[QueryAgent] = None
_agent_collection: Optional[str] = None

def get_agent(collection_name: Optional[str] = None) -> QueryAgent:
    """Return a cached QueryAgent for the given collection."""
    global _agent, _agent_collection

    name = collection_name or COLLECTION_NAME

    if _agent is None or _agent_collection != name:
        client = get_client()
        _agent = QueryAgent(client=client, collections=[name])
        _agent_collection = name
    return _agent

def ask(question: str, collection_name: Optional[str] = None) -> dict:
    """Run a natural-language query and return a serializable dict."""
    agent = get_agent(collection_name)
    resp = agent.ask(question)

    # Extract whatever the agent exposes; adapt to your version
    answer = getattr(resp, "final_answer", None) or str(resp)

    sources = []
    for src in getattr(resp, "sources", []) or []:
        if isinstance(src, dict):
            sources.append(src)
        else:
            sources.append({
                "text": getattr(src, "text", "") or "",
                "source": getattr(src, "source", None),
                "page": getattr(src, "page", None),
                "chunk_id": getattr(src, "chunk_id", None),
            })

    return {"answer": answer, "sources": sources}