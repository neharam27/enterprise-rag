from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION  = os.getenv("QDRANT_COLLECTION_NAME", "enterprise_docs")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
client     = QdrantClient(url=QDRANT_URL)

def dense_search(query: str, user_role: str, top_k: int = 20) -> list[dict]:
    """
    Semantic vector search with RBAC filtering.
    Only returns chunks the user's role is allowed to see.
    """
    query_vector = embeddings.embed_query(query)

    # RBAC filter — Qdrant only returns docs where allowed_roles contains user_role
    rbac_filter = Filter(
        must=[
            FieldCondition(
                key="allowed_roles",
                match=MatchAny(any=[user_role])
            )
        ]
    )

    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        query_filter=rbac_filter,
        limit=top_k,
        with_payload=True
    )

    return [
        {
            "chunk_id":       r.payload.get("chunk_id"),
            "content":        r.payload.get("content"),
            "parent_content": r.payload.get("parent_content", ""),
            "source_file":    r.payload.get("source_file"),
            "page_number":    r.payload.get("page_number"),
            "chunk_type":     r.payload.get("chunk_type"),
            "score":          r.score,
            "retriever":      "dense"
        }
        for r in results
    ]