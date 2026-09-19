import os
import logging
from fastmcp import FastMCP
from weaviate_client import (
    get_agent,
    search_collection,
    get_client,
    COLLECTION_NAME,
)
from formatter import format_answer, format_search_results

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialize the FastMCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("SOC 101 Sociology Assistant")

# ---------------------------------------------------------------------------
# 1. MCP Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def ask_sociology_question(question: str) -> str:
    """
    Query the Introduction to Sociology course materials to answer natural-language questions.
    Returns a grounded answer with module, heading, and page citations.
    """
    if not question or not question.strip():
        return "Error: question cannot be empty."
    if len(question) > 2000:
        return "Error: question is too long (max 2000 characters)."

    try:
        agent = get_agent()
        result = agent.ask(question)

        sources = []
        for s in getattr(result, "sources", []) or []:
            props = getattr(s, "properties", None) or {}
            sources.append(
                {
                    "heading": props.get("heading"),
                    "module_title": props.get("module_title"),
                    "page": props.get("page"),
                    "source": props.get("source"),
                    "chunk_id": props.get("chunk_id"),
                }
            )

        raw_answer = getattr(result, "final_answer", None) or str(result)
        return format_answer(raw_answer, sources, include_sources=True)

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return "An internal error occurred while generating the answer."


@mcp.tool()
def search_sociology_passages(
    query: str, limit: int = 5, alpha: float = 0.7
) -> str:
    """
    Perform a direct hybrid search against the SOC 101 collection without LLM answer generation.
    Returns ranked passages with source citations.
    """
    if not query or not query.strip():
        return "Error: query cannot be empty."
    if len(query) > 500:
        return "Error: query is too long (max 500 characters)."
    if not (1 <= limit <= 30):
        return "Error: limit must be between 1 and 30."
    if not (0.0 <= alpha <= 1.0):
        return "Error: alpha must be between 0.0 and 1.0."

    try:
        hits = search_collection(query, limit=limit, alpha=alpha)
        return format_search_results(hits)
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return "An internal error occurred while searching."


# ---------------------------------------------------------------------------
# 2. MCP Resources
# ---------------------------------------------------------------------------
@mcp.resource("system://status")
def get_system_status() -> str:
    """Check the connection status of the Weaviate cluster and target collection."""
    try:
        client = get_client()
        is_ready = client.is_ready()
        has_collection = client.collections.exists(COLLECTION_NAME)
        return (
            f"System Status: ONLINE\n"
            f"Weaviate Connected: {is_ready}\n"
            f"Target Collection ('{COLLECTION_NAME}'): "
            f"{'Found' if has_collection else 'Missing'}"
        )
    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        return "System Status: OFFLINE\nUnable to reach Weaviate."


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
