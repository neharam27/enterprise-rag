from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
from typing import Optional
import uuid
from .pdf_parser import ParsedElement

@dataclass
class Chunk:
    chunk_id: str
    parent_id: Optional[str]    # None = this IS a parent chunk
    content: str
    chunk_type: str             # "parent" | "child" | "table" | "image"
    page_number: int
    source_file: str
    metadata: dict

def chunk_elements(elements: list[ParsedElement]) -> list[Chunk]:
    chunks = []

    # Splitter for parent chunks — large, preserves context
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Splitter for child chunks — small, precise retrieval
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for element in elements:

        # Tables go in as-is — don't chunk them, they lose meaning
        if element.type == "table":
            chunk_id = str(uuid.uuid4())
            chunks.append(Chunk(
                chunk_id=chunk_id,
                parent_id=None,
                content=element.content,
                chunk_type="table",
                page_number=element.page_number,
                source_file=element.source_file,
                metadata={**element.metadata, "element_type": "table"}
            ))

        # Images go in as-is — content will be caption text later
        elif element.type == "image":
            chunk_id = str(uuid.uuid4())
            chunks.append(Chunk(
                chunk_id=chunk_id,
                parent_id=None,
                content=element.content,
                chunk_type="image",
                page_number=element.page_number,
                source_file=element.source_file,
                metadata={**element.metadata, "element_type": "image"}
            ))

        # Text — apply parent-child chunking
        elif element.type == "text":
            parent_texts = parent_splitter.split_text(element.content)

            for parent_text in parent_texts:
                parent_id = str(uuid.uuid4())

                # Store the parent chunk (large context)
                chunks.append(Chunk(
                    chunk_id=parent_id,
                    parent_id=None,
                    content=parent_text,
                    chunk_type="parent",
                    page_number=element.page_number,
                    source_file=element.source_file,
                    metadata={"element_type": "text"}
                ))

                # Split parent into children (small, for retrieval)
                child_texts = child_splitter.split_text(parent_text)

                for child_text in child_texts:
                    child_id = str(uuid.uuid4())
                    chunks.append(Chunk(
                        chunk_id=child_id,
                        parent_id=parent_id,   # links back to parent
                        content=child_text,
                        chunk_type="child",
                        page_number=element.page_number,
                        source_file=element.source_file,
                        metadata={"element_type": "text"}
                    ))

    parents = sum(1 for c in chunks if c.chunk_type == "parent")
    children = sum(1 for c in chunks if c.chunk_type == "child")
    tables = sum(1 for c in chunks if c.chunk_type == "table")
    images = sum(1 for c in chunks if c.chunk_type == "image")
    print(f"  Chunked: {parents} parents → {children} children | {tables} tables | {images} images")
    return chunks