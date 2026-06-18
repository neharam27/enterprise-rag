from .dense_retriever import dense_search
from .sparse_retriever import sparse_search
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

load_dotenv()

# RRF constant — 60 is standard, higher = smoother rank fusion
RRF_K = 60

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict]
) -> list[dict]:
    """
    Fuses dense + sparse result lists using Reciprocal Rank Fusion.

    Formula: RRF(d) = Σ 1 / (k + rank(d))
    Each document gets a score from both lists; scores are summed.
    Documents appearing in both lists get a big boost.
    """
    rrf_scores = {}   # chunk_id → rrf_score
    chunk_map  = {}   # chunk_id → chunk dict

    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank)
        chunk_map[cid] = result

    for rank, result in enumerate(sparse_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank)
        chunk_map[cid] = result

    # Sort by fused RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused = []
    for cid in sorted_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = round(rrf_scores[cid], 6)
        fused.append(chunk)

    return fused

def rerank_with_llm(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder style reranking using the local Ollama LLM.
    Asks the LLM to score each chunk's relevance to the query (0-10).
    Falls back gracefully if scoring fails.

    Note: For production, swap this with a dedicated cross-encoder like
    'cross-encoder/ms-marco-MiniLM-L-6-v2' via sentence-transformers for speed.
    """
    llm = OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL", "llama3.1"))

    candidates = chunks[:15]  # only rerank top 15 from RRF
    scored = []

    for chunk in candidates:
        prompt = f"""Rate how relevant this passage is to answering the query.
Return ONLY a number from 0 to 10. No explanation.

Query: {query}

Passage: {chunk['content'][:500]}

Relevance score (0-10):"""

        try:
            response = llm.invoke(prompt).strip()
            # Extract first number found in response
            import re
            numbers = re.findall(r'\b([0-9]|10)\b', response)
            score = float(numbers[0]) if numbers else 5.0
        except Exception:
            score = 5.0

        scored.append({**chunk, "rerank_score": score})

    # Sort by rerank score, then return top_k
    reranked = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

def context_expansion(chunk: dict) -> str:
    """
    Parent-child context expansion:
    If a child chunk was retrieved, return its parent content for the LLM.
    This gives the LLM full context, not just the small retrieved snippet.
    """
    if chunk.get("parent_content"):
        return chunk["parent_content"]
    return chunk["content"]

def hybrid_search(
    query: str,
    user_role: str,
    top_k: int = 5,
    use_reranking: bool = True
) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
    1. Dense search (semantic)
    2. Sparse search (BM25 keyword)
    3. RRF fusion
    4. LLM reranking
    5. Parent context expansion
    """
    print(f"\n  [Dense search...]")
    dense_results  = dense_search(query, user_role, top_k=20)
    print(f"  Dense: {len(dense_results)} results")

    print(f"  [Sparse search...]")
    sparse_results = sparse_search(query, user_role, top_k=20)
    print(f"  Sparse: {len(sparse_results)} results")

    print(f"  [RRF fusion...]")
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    print(f"  Fused: {len(fused)} unique results")

    if use_reranking:
        print(f"  [Reranking top 15...]")
        final = rerank_with_llm(query, fused, top_k=top_k)
    else:
        final = fused[:top_k]

    # Expand context using parent chunks
    for chunk in final:
        chunk["context_for_llm"] = context_expansion(chunk)

    return final