import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import base64

@dataclass
class ParsedElement:
    type: str          # "text" | "table" | "image"
    content: str       # raw text, table as markdown, or image caption placeholder
    page_number: int
    source_file: str
    metadata: dict

def parse_pdf(pdf_path: str) -> list[ParsedElement]:
    path = Path(pdf_path)
    elements = []

    # --- Extract text + tables with pdfplumber ---
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # Extract tables first (so we don't double-extract that text)
            tables = page.extract_tables()
            table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

            for table in tables:
                if not table:
                    continue
                # Convert table to markdown string
                md_rows = []
                for i, row in enumerate(table):
                    cleaned = [cell or "" for cell in row]
                    md_rows.append("| " + " | ".join(cleaned) + " |")
                    if i == 0:
                        md_rows.append("| " + " | ".join(["---"] * len(cleaned)) + " |")
                table_md = "\n".join(md_rows)

                elements.append(ParsedElement(
                    type="table",
                    content=table_md,
                    page_number=page_num,
                    source_file=path.name,
                    metadata={"has_table": True}
                ))

            # Extract text (excluding table regions)
            text = page.extract_text()
            if text and text.strip():
                elements.append(ParsedElement(
                    type="text",
                    content=text.strip(),
                    page_number=page_num,
                    source_file=path.name,
                    metadata={}
                ))

    # --- Extract images with PyMuPDF ---
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Save image to disk for later captioning
            img_dir = Path("data/processed/images")
            img_dir.mkdir(parents=True, exist_ok=True)
            img_filename = f"{path.stem}_p{page_num+1}_img{img_index}.png"
            img_path = img_dir / img_filename
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            elements.append(ParsedElement(
                type="image",
                content=f"[IMAGE: {img_filename}]",  # placeholder, captioned later
                page_number=page_num + 1,
                source_file=path.name,
                metadata={"image_path": str(img_path)}
            ))

    doc.close()
    print(f"  Parsed {path.name}: {len(elements)} elements extracted")
    return elements