from langchain_ollama import OllamaLLM
import os
import re

llm = OllamaLLM(
    model=os.getenv("OLLAMA_LLM_MODEL", "llama3.1"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

GROUNDED_ANSWER_PROMPT = """You are a precise enterprise document assistant. Answer the question using ONLY the context below.

Rules:
- Cite every factual claim using [Source N] notation, where N matches the source number below.
- If the answer comes from a table, explicitly mention that it's from a table.
- If the context does NOT contain enough information to answer, respond with exactly: "INSUFFICIENT_CONTEXT" — nothing else.
- Do not use outside knowledge. Do not guess. Do not fill gaps with assumptions.
- Be concise — answer in 1-3 sentences unless the question requires a list or table.

Context:
{context}

Question: {question}

Answer:"""

def build_context_blocks(results: list[dict]) -> tuple[str, dict]:
    """
    Builds numbered context blocks for the prompt and a lookup
    so we can map [Source N] back to the actual chunk metadata.
    """
    blocks = []
    source_map = {}

    for i, r in enumerate(results, 1):
        tag = "[TABLE DATA]" if r["chunk_type"] == "table" else ""
        block = f"[Source {i}] {tag} (from {r['source_file']}, page {r['page_number']})\n{r['context_for_llm']}"
        blocks.append(block)
        source_map[i] = r

    return "\n\n".join(blocks), source_map

def compute_confidence(results: list[dict], grounded: bool) -> str:
    """
    Heuristic confidence score based on retrieval signal strength.
    Not a substitute for real eval, but gives a useful signal to the user.
    """
    if not grounded or not results:
        return "low"

    top_score = results[0].get("rerank_score", results[0].get("rrf_score", 0))

    # rerank_score is on a 0-10 scale; rrf_score is much smaller (~0.01-0.05 range)
    if "rerank_score" in results[0]:
        if top_score >= 8:
            return "high"
        elif top_score >= 5:
            return "medium"
        else:
            return "low"
    else:
        if top_score >= 0.03:
            return "high"
        elif top_score >= 0.015:
            return "medium"
        else:
            return "low"

def extract_cited_sources(answer: str, source_map: dict) -> set[int]:
    """Parses which [Source N] tags actually appear in the generated answer."""
    cited = set(int(n) for n in re.findall(r'\[Source (\d+)\]', answer))
    return {n for n in cited if n in source_map}

def generate_grounded_answer(query: str, results: list[dict]) -> dict:
    """
    Full answer generation with grounding check, citation extraction,
    and confidence scoring.
    """
    if not results:
        return {
            "answer": "No relevant documents were found for your query.",
            "sources": [],
            "confidence": "low",
            "grounded": False
        }

    context, source_map = build_context_blocks(results)
    prompt = GROUNDED_ANSWER_PROMPT.format(context=context, question=query)

    raw_answer = llm.invoke(prompt).strip()

    if raw_answer == "INSUFFICIENT_CONTEXT" or "INSUFFICIENT_CONTEXT" in raw_answer:
        return {
            "answer": "The retrieved documents don't contain enough information to answer this question confidently.",
            "sources": [],
            "confidence": "low",
            "grounded": False
        }

    cited_ids = extract_cited_sources(raw_answer, source_map)
    grounded = len(cited_ids) > 0

    # Only return sources that were ACTUALLY cited, not all retrieved chunks
    cited_sources = [
        {
            "source_id": sid,
            "source_file": source_map[sid]["source_file"],
            "page_number": source_map[sid]["page_number"],
            "chunk_type": source_map[sid]["chunk_type"],
            "content": source_map[sid]["content"][:300],
            "relevance_score": source_map[sid].get("rerank_score", source_map[sid].get("rrf_score", 0))
        }
        for sid in sorted(cited_ids)
    ]

    confidence = compute_confidence(results, grounded)

    return {
        "answer": raw_answer,
        "sources": cited_sources,
        "confidence": confidence,
        "grounded": grounded
    }