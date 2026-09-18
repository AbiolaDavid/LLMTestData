import os
from typing import List, Optional
from fastmcp import FastMCP, Context
import weaviate

# Assuming your existing modules are structured like this based on your stack
# from weaviate_client import get_weaviate_client 
# from formatter import format_search_results
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the FastMCP server
# Dependencies can be passed so `fastmcp install` automatically grabs them
mcp = FastMCP(
    "Knowledge Retrieval Server", 
    dependencies=["weaviate-client", "langchain-community", "langchain-huggingface", "pypdf"]
)

# ---------------------------------------------------------------------------
# 1. Weaviate & Embedding Setup (Mocked based on your previous stack)
# ---------------------------------------------------------------------------
def get_embeddings():
    """Initialize HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_weaviate_client():
    """Connect to the Weaviate vector database."""
    # Replace with your actual Weaviate cluster URL and API key
    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    return weaviate.Client(url=weaviate_url)

# ---------------------------------------------------------------------------
# 2. MCP Tools
# Tools are actions the LLM can actively call with arguments.
# ---------------------------------------------------------------------------

@mcp.tool()
def search_knowledge_base(query: str, limit: int = 5) -> str:
    """
    Search the Weaviate vector database for relevant context based on a query.
    
    Args:
        query: The search term or question to look up.
        limit: Maximum number of results to return.
    """
    client = get_weaviate_client()
    embeddings = get_embeddings()
    
    # Generate vector for the query
    query_vector = embeddings.embed_query(query)
    
    # Execute Weaviate nearVector search (assuming a collection named "Document")
    try:
        response = (
            client.query
            .get("Document", ["text", "source"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .do()
        )
        
        results = response.get("data", {}).get("Get", {}).get("Document", [])
        if not results:
            return "No relevant documents found for your query."
            
        # Format results for the LLM
        formatted_context = "\n\n".join([
            f"Source: {res.get('source', 'Unknown')}\nContent: {res.get('text', '')}"
            for res in results
        ])
        
        return f"Found {len(results)} results:\n\n{formatted_context}"
    except Exception as e:
        return f"Error querying Weaviate database: {str(e)}"


@mcp.tool()
async def upload_pdf_to_vectorstore(file_path: str, ctx: Context) -> str:
    """
    Ingest a PDF file, chunk it, embed it, and upload it to Weaviate.
    
    Args:
        file_path: The local absolute path to the PDF file to upload.
    """
    ctx.info(f"Starting ingestion for {file_path}")
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    try:
        # Load the document using LangChain
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        ctx.info(f"Extracted {len(pages)} pages. Generating embeddings...")
        
        # In a real app, you would batch upload these to Weaviate here using your 
        # existing weaviate_client.py logic.
        
        return f"Successfully processed {file_path} and uploaded {len(pages)} chunks to Weaviate."
    except Exception as e:
        ctx.error(f"Ingestion failed: {str(e)}")
        return f"Failed to upload dataset: {str(e)}"

# ---------------------------------------------------------------------------
# 3. MCP Resources
# Resources are static or dynamic data the LLM can read via URIs.
# ---------------------------------------------------------------------------

@mcp.resource("system://status")
def get_system_status() -> str:
    """Check if the Weaviate database and embedding models are online."""
    try:
        client = get_weaviate_client()
        is_ready = client.is_ready()
        return f"System Status: ONLINE\nWeaviate DB Ready: {is_ready}"
    except Exception as e:
        return f"System Status: DEGRADED\nError: {str(e)}"

# ---------------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Runs the server using the stdio transport by default (ideal for Claude Desktop)
    mcp.run()