import os
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client() -> weaviate.WeaviateClient:
    """Singleton Weaviate Cloud client."""
    global _client
    if _client is None or not _client.is_ready():
        url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")
        if not url or not api_key:
            raise RuntimeError("WEAVIATE_URL and WEAVIATE_API_KEY must be set")

        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=60, query=240, insert=240)
            ),
        )
    return _client

def close_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None