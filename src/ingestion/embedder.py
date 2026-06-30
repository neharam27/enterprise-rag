from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PayloadSchemaType
)
from .chunker import Chunk
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "enterprise_docs")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
client     = QdrantClient(url=QDRANT_URL)

def init_collection(vector_size: int = 768):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"  Created Qdrant collection: {COLLECTION}")
    else:
        print(f"  Collection '{COLLECTION}' already exists")

def embed_and_store(chunks: list[Chunk], allowed_roles: list[str] = None):
    if allowed_roles is None:
        allowed_roles = ["admin", "analyst", "viewer"]

    # Only embed child chunks + tables + images (not parents — parents are for context only)
    indexable = [c for c in chunks if c.chunk_type in ("child", "table", "image")]
    # Keep parents in a lookup dict for context retrieval
    parent_map = {c.chunk_id: c.content for c in chunks if c.chunk_type == "parent"}

    print(f"  Embedding {len(indexable)} chunks...")

    batch_size = 32
    points = []

    for i in range(0, len(indexable), batch_size):
        batch = indexable[i:i+batch_size]
        texts = [c.content for c in batch]
        vectors = embeddings.embed_documents(texts)

        for chunk, vector in zip(batch, vectors):
            parent_content = parent_map.get(chunk.parent_id, "") if chunk.parent_id else ""

            payload = {
                "chunk_id":      chunk.chunk_id,
                "parent_id":     chunk.parent_id,
                "parent_content": parent_content,   # stored for context expansion
                "content":       chunk.content,
                "chunk_type":    chunk.chunk_type,
                "page_number":   chunk.page_number,
                "source_file":   chunk.source_file,
                "allowed_roles": allowed_roles,      # RBAC field
                **chunk.metadata
            }
            points.append(PointStruct(id=hash(chunk.chunk_id) % (2**63), vector=vector, payload=payload))

        print(f"    Embedded batch {i//batch_size + 1}/{(len(indexable)-1)//batch_size + 1}")

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"  Stored {len(points)} vectors in Qdrant")