from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil

from src.rbac import init_db, create_user, authenticate_user, create_access_token, get_current_user, require_role
from src.api.schemas import (
    LoginRequest, TokenResponse, QueryRequest, QueryResponse,
    SourceChunk, CreateUserRequest
)
from src.retrieval import hybrid_search, generate_grounded_answer

app = FastAPI(title="Enterprise Document Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()
    create_user("admin1", "adminpass123", "admin")
    create_user("analyst1", "analystpass123", "analyst")
    create_user("viewer1", "viewerpass123", "viewer")

# ---------- AUTH ----------

@app.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(username=user["username"], role=user["role"])
    return TokenResponse(access_token=token, role=user["role"])

@app.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return current_user

# ---------- ADMIN ----------

@app.post("/admin/create-user")
def admin_create_user(
    request: CreateUserRequest,
    current_user: dict = Depends(require_role("admin"))
):
    create_user(request.username, request.password, request.role)
    return {"message": f"User '{request.username}' created with role '{request.role}'"}

@app.post("/admin/upload")
async def admin_upload(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_role("admin"))
):
    upload_dir = Path("data/raw")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "message": f"File '{file.filename}' uploaded. Run ingestion separately to index it.",
        "path": str(dest)
    }

# ---------- QUERY ----------

@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    results = hybrid_search(
        query=request.query,
        user_role=current_user["role"],
        top_k=request.top_k,
        use_reranking=request.use_reranking
    )

    result = generate_grounded_answer(request.query, results)

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        confidence=result["confidence"],
        grounded=result["grounded"]
    )

@app.get("/health")
def health():
    return {"status": "ok"}