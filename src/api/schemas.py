from pydantic import BaseModel
from typing import Optional

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True

class SourceChunk(BaseModel):
    source_id: int          # NEW — matches [Source N] in the answer
    source_file: str
    page_number: int
    chunk_type: str
    content: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str          # NEW — "high" | "medium" | "low"
    grounded: bool           # NEW — did the LLM actually use the context?

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str