from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
import os
import re
from dotenv import load_dotenv

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "enterprise_docs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

client = QdrantClient(url=QDRANT_URL)

def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r'\w+', text.lower())

def build_bm25_index(user_role: str):
    """
    Fetches all allowed chunks from Qdrant and builds a BM25 index.
    Returns (bm25, all_chunks) tuple.
    """
    rbac_filter = Filter(
        must=[
            FieldCondition(
                key="allowed_roles",
                match=MatchAny(any=[user_role])
            )
        ]
    )

    # Scroll through all matching points
    all_chunks = []
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=rbac_filter,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        all_chunks.extend(results)
        if offset is None:
            break

    if not all_chunks:
        return None, []

    corpus = [tokenize(r.payload.get("content", "")) for r in all_chunks]
    bm25 = BM25Okapi(corpus)
    return bm25, all_chunks

def sparse_search(query: str, user_role: str, top_k: int = 20) -> list[dict]:
    """
    BM25 keyword search with RBAC — only indexes chunks the role can access.
    """
    bm25, all_chunks = build_bm25_index(user_role)

    if not bm25 or not all_chunks:
        return []

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its BM25 score and sort
    scored = sorted(
        zip(scores, all_chunks),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [
        {
            "chunk_id":       chunk.payload.get("chunk_id"),
            "content":        chunk.payload.get("content"),
            "parent_content": chunk.payload.get("parent_content", ""),
            "source_file":    chunk.payload.get("source_file"),
            "page_number":    chunk.payload.get("page_number"),
            "chunk_type":     chunk.payload.get("chunk_type"),
            "score":          float(score),
            "retriever":      "sparse"
        }
        for score, chunk in scored
        if score > 0  # BM25 score of 0 = no keyword match at all
    ]