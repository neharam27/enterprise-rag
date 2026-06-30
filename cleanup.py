from qdrant_client import QdrantClient
client = QdrantClient(url="http://qdrant:6333")
client.delete_collection("enterprise_docs")
print("Deleted incomplete collection")