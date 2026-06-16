import spacy

nlp = spacy.load("en_core_web_sm")

ENTITY_TYPES = {
    "ORG":      "organizations",
    "PERSON":   "people",
    "GPE":      "locations",
    "MONEY":    "monetary_values",
    "DATE":     "dates",
    "PERCENT":  "percentages",
    "PRODUCT":  "products",
}

def extract_entities(text: str) -> dict:
    doc = nlp(text[:10000])  # spaCy limit safety
    entities = {label: [] for label in ENTITY_TYPES.values()}

    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            label = ENTITY_TYPES[ent.label_]
            value = ent.text.strip()
            if value and value not in entities[label]:
                entities[label].append(value)

    # Flatten to only non-empty fields
    return {k: v for k, v in entities.items() if v}

def enrich_chunks_with_ner(chunks):
    for chunk in chunks:
        if chunk.chunk_type in ("child", "table"):
            entities = extract_entities(chunk.content)
            chunk.metadata.update(entities)
    print(f"  NER complete on {len(chunks)} chunks")
    return chunks