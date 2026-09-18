import os
from fastmcp import FastMCP
from weaviate_client import (
    get_agent,
    search_collection,
    get_client,
    COLLECTION_NAME,
)
from formatter import format_answer, format_search_results

# Initialize the FastMCP server
mcp = FastMCP("SOC 101 Sociology Assistant")

# ---------------------------------------------------------------------------
# 1. MCP Tools (Actions the AI client can execute)
# ---------------------------------------------------------------------------
@mcp.tool()
def ask_sociology_question(question: str) -> str:
    """
    Query the Introduction to Sociology course materials to answer natural-language questions.
    Returns a grounded answer with module, heading, and page citations.
    """
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
        return f"Agent error: {e}"

@mcp.tool()
def search_sociology_passages(
    query: str, limit: int = 5, alpha: float = 0.7
) -> str:
    """
    Perform a direct hybrid search against the SOC 101 collection without LLM answer generation.
    Returns ranked passages with source citations.
    """
    try:
        hits = search_collection(query, limit=limit, alpha=alpha)
        return format_search_results(hits)
    except Exception as e:
        return f"Search error: {e}"

# ---------------------------------------------------------------------------
# 2. MCP Resources (Context the AI client can inspect)
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
            f"Target Collection ('{COLLECTION_NAME}'): {'Found' if has_collection else 'Missing'}"
        )
    except Exception as e:
        return f"System Status: OFFLINE\nError: {e}"

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # FastMCP handles CLI arguments and transport setup automatically
    mcp.run()
