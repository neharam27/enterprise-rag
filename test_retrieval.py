from src.retrieval import hybrid_search

query     = "What was Apple's total revenue?"
user_role = "analyst"

print(f"\nQuery: {query}")
print(f"Role:  {user_role}\n")

results = hybrid_search(query, user_role, top_k=3, use_reranking=True)

print("\n=== TOP RESULTS ===\n")
for i, r in enumerate(results, 1):
    print(f"[{i}] Source: {r['source_file']} | Page: {r['page_number']}")
    print(f"    RRF Score: {r.get('rrf_score', 'N/A')} | Rerank: {r.get('rerank_score', 'N/A')}")
    print(f"    Retriever: {r['retriever']} | Type: {r['chunk_type']}")
    print(f"    Content: {r['content'][:200]}...")
    print()