import sys
from pathlib import Path
from src.ingestion import (
    parse_pdf, chunk_elements,
    enrich_chunks_with_ner, init_collection, embed_and_store
)

def ingest(pdf_path: str, roles: list[str] = None):
    if roles is None:
        roles = ["admin", "analyst", "viewer"]

    print(f"\n Ingesting: {pdf_path}")

    print("\n[1/4] Parsing PDF...")
    elements = parse_pdf(pdf_path)

    print("\n[2/4] Chunking...")
    chunks = chunk_elements(elements)

    print("\n[3/4] Running NER...")
    chunks = enrich_chunks_with_ner(chunks)

    print("\n[4/4] Embedding and storing in Qdrant...")
    init_collection(vector_size=768)
    embed_and_store(chunks, allowed_roles=roles)

    print(f"\n Done! {pdf_path} is fully ingested and ready to query.\n")

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.pdf"
    ingest(pdf_path)