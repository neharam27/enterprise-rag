from langchain_ollama import OllamaEmbeddings, OllamaLLM
from qdrant_client import QdrantClient

# Test embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
result = embeddings.embed_query("test sentence")
print(f"Embedding works — vector size: {len(result)}")

# Test LLM
llm = OllamaLLM(model="llama3.1")
response = llm.invoke("Say hello in one sentence.")
print(f"LLM works — response: {response}")

# Test Qdrant
client = QdrantClient(url="http://localhost:6333")
print(f"Qdrant works — collections: {client.get_collections()}")

print("\n All systems go. Ready to build.")
