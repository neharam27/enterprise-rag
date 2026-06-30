# 🏢 Enterprise Document Intelligence Platform

A productionstyle **Retrieval-Augmented Generation (RAG)** system for querying complex enterprise documents financial filings, contracts, technical reports  with **role based access control**, **hybrid retrieval**, and **multimodal document understanding**.

Built entirely with **LangChain** and runs **100% locally** via **Ollama** zero API cost, zero data leaving your machine. Fully containerized with **Docker Compose** for one command startup.

> Demo dataset: Apple Inc.'s FY2024 10-K filing (SEC EDGAR, public data).

---

## Why this project

Most RAG tutorials stop at "upload a PDF, ask a question." This project goes further by solving the problems that actually show up in real enterprise deployments:

- Documents have **structure** (nested sections, tables, charts) — naive chunking destroys it
- Different users need **different access** to the same document store — security can't be an afterthought
- A single retrieval method (just embeddings, or just keyword search) **misses things** the other would catch
- LLMs **hallucinate** when given irrelevant context instead of admitting they don't know

Every technique in this system exists to solve one of these four problems, not as a checklist of buzzwords.

---

## Architecture

```
                                ┌─────────────────────┐
                                │   Streamlit UI       │
                                │  (Chat + Admin)      │
                                └──────────┬───────────┘
                                           │ JWT-authenticated REST
                                           ▼
                                ┌─────────────────────┐
                                │   FastAPI Backend    │
                                │  /login /query /admin│
                                └──────────┬───────────┘
                  ┌────────────────────────┼────────────────────────┐
                  ▼                        ▼                        ▼
         ┌────────────────┐     ┌──────────────────┐      ┌──────────────────┐
         │  SQLite (RBAC)  │     │  Hybrid Retrieval │      │  Ollama (local)  │
         │  users + roles  │     │  Qdrant + BM25    │      │  llama3.1        │
         │  bcrypt hashes  │     │  RRF + Reranking  │      │  nomic-embed-text│
         └────────────────┘     └──────────────────┘      └──────────────────┘
                                           ▲
                                           │
                                ┌─────────────────────┐
                                │  Ingestion Pipeline  │
                                │  pdfplumber + PyMuPDF │
                                │  Parent-Child Chunking│
                                │  spaCy NER            │
                                └─────────────────────┘
```

---

## How a query actually flows through the system

1. **Auth** — user logs in, receives a JWT encoding their username and role (`admin` / `analyst` / `viewer`)
2. **Dense retrieval** — query is embedded (`nomic-embed-text`) and matched against Qdrant, filtered server-side so only documents the user's role can access are even considered
3. **Sparse retrieval** — the same query runs through BM25 keyword matching over the same RBAC-filtered document set
4. **RRF fusion** — the two ranked lists are merged using Reciprocal Rank Fusion, which rewards chunks that rank highly in *either or both* lists without needing to manually tune a dense/sparse weighting
5. **Reranking** — the top ~15 fused candidates are individually scored for relevance by the LLM, and only the top-k survive
6. **Context expansion** — retrieved *child* chunks (small, precise) are swapped for their *parent* chunk (large, full context) before being sent to the LLM  this is the parent-child chunking strategy paying off
7. **Grounded generation** — the LLM answers strictly from the provided context, citing `[Source N]` tags; if the context doesn't support an answer, it must emit a sentinel value instead of guessing
8. **Citation verification** — the backend parses which `[Source N]` tags the LLM actually used and returns *only those* as evidence, not every chunk that was retrieved  closing the gap between "what was fetched" and "what was actually relied on"

---

## Key engineering decisions

### Parent-child + recursive chunking
Financial and legal documents have deep structural hierarchy. Recursive chunking respects natural boundaries (paragraphs → sentences) rather than blindly cutting at a fixed character count. Parent-child chunking means retrieval operates on small, precise child chunks (good for matching), while the LLM is given the larger parent chunk (good for context)  solving the classic RAG tradeoff between retrieval precision and generation context.

### Table and image extraction
`pdfplumber` extracts tables as structured markdown rather than letting them collapse into unreadable run-on text. This is the difference between a system that can answer "how much revenue came from iPad sales?" correctly versus one that can't parse a table at all. Images are extracted via PyMuPDF and stored with placeholder captions for future vision-model captioning.

### Hybrid search instead of pure embeddings
Dense embeddings excel at semantic similarity but can miss exact terminology (`"EBITDA"`, specific clause numbers). BM25 excels at exact-term matching but misses paraphrased questions. Running both and fusing with RRF captures both failure modes' complements without manual weight tuning.

### RBAC enforced at the retrieval layer, not just the API layer
Role filtering happens as a Qdrant payload filter *during* the vector search itself not as a post-hoc check on returned results. This means a user's role is structurally incapable of retrieving documents outside their permission, rather than relying on an application-layer check that could be bypassed.

### Explicit hallucination refusal
The LLM is instructed to emit `INSUFFICIENT_CONTEXT` when retrieved chunks don't actually answer the question, and the backend checks for this explicitly. Combined with citation verification (step 8 above), the system can confidently say "I don't know" instead of confidently making something up validated directly against out-of-scope questions like "What is the capital of France?"

### Fully local inference via Ollama
No OpenAI/Anthropic API calls anywhere in the pipeline. Every embedding, generation, and reranking call runs on local hardware via Ollama (`llama3.1` for generation/reranking, `nomic-embed-text` for embeddings). This makes the system genuinely deployable in regulated or air-gapped enterprise environments where sending documents to a third-party API isn't acceptable and it's free to run indefinitely.

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain (`langchain-ollama`, `langchain-community`) |
| LLM + Embeddings | Ollama — `llama3.1` (8B), `nomic-embed-text` |
| Vector store | Qdrant (with payload-based RBAC filtering) |
| Sparse retrieval | `rank_bm25` |
| Entity extraction | spaCy (`en_core_web_sm`) |
| PDF parsing | `pdfplumber` (tables), PyMuPDF (images) |
| Backend | FastAPI, JWT (`python-jose`), `bcrypt` |
| User storage | SQLite |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |

---

## Project structure

```
enterprise-rag/
├── src/
│   ├── ingestion/        # PDF parsing, chunking, NER, embedding
│   │   ├── pdf_parser.py
│   │   ├── chunker.py
│   │   ├── ner_extractor.py
│   │   └── embedder.py
│   ├── retrieval/         # hybrid search, RRF, reranking, generation
│   │   ├── dense_retriever.py
│   │   ├── sparse_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── answer_generator.py
│   ├── rbac/              # auth, JWT, user database
│   │   ├── database.py
│   │   └── auth.py
│   └── api/                # FastAPI app
│       ├── main.py
│       └── schemas.py
├── frontend/                # Streamlit UI
│   ├── app.py
│   └── api_client.py
├── data/
│   ├── raw/                 # uploaded PDFs
│   └── processed/            # extracted images
├── ingest.py                  # CLI ingestion entry point
├── docker-compose.yml
├── Dockerfile.backend
├── frontend/Dockerfile.frontend
└── requirements.txt
```

---

## Running it locally

### Prerequisites
- Docker Desktop
- [Ollama](https://ollama.com) installed natively on the host machine (not containerized Docker-to-GPU passthrough for LLM inference adds complexity without benefit for local dev)

### 1. Pull the required models
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 2. Start the full stack
```bash
docker-compose up --build
```
This starts Qdrant, the FastAPI backend, and the Streamlit frontend, all networked together. The backend reaches your natively-running Ollama instance via `host.docker.internal`.

### 3. Ingest a document
```bash
docker-compose exec backend python ingest.py data/raw/your_document.pdf
```

### 4. Open the app
```
http://localhost:8501
```

### Demo accounts (seeded automatically on first startup)
| Username | Password | Role |
|---|---|---|
| `admin1` | `adminpass123` | admin |
| `analyst1` | `analystpass123` | analyst |
| `viewer1` | `viewerpass123` | viewer |

### Stopping
```bash
docker-compose down
```
Data persists across restarts Qdrant vectors live in a named Docker volume, and the SQLite user database lives in `./data`, mounted from the host.

---

## Known limitations and future improvements

This section is deliberately included knowing a system's limitations is as important as building its features.

- **LLM-as-reranker has run-to-run variance.** Using `llama3.1` itself to score chunk relevance (instead of a dedicated cross-encoder) is cost-free but not fully deterministic — the same query can occasionally surface a different (but still valid) top result across runs, particularly when multiple chunks contain similarly relevant data (e.g., the same financial metric across different fiscal years). A production system would swap this for `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` for speed and determinism.
- **No temporal disambiguation in financial tables.** When a table contains multiple fiscal years' data, the system doesn't always make explicit which year it's reporting unless the question specifies one. Worth adding explicit year extraction logic for financial documents specifically.
- **Image understanding is a placeholder.** Images are extracted and stored, but not yet captioned via a vision model they're not currently searchable by content. The architecture supports adding this (e.g., via GPT-4o or LLaVA captioning at ingestion time).
- **Token expiry has no graceful UX.** When a JWT expires mid-session, the user sees a raw "Invalid or expired token" error rather than being automatically redirected to re-login.
- **Single-node Qdrant.** No replication or sharding appropriate for a portfolio/demo project, not for production-scale document volumes.
- **No public deployment.** This runs locally by design Ollama requires meaningful compute that isn't available on free tier cloud hosting. A cloud-deployed version would need either a paid GPU instance or swapping Ollama for a hosted LLM API for that deployment specifically.

---

## What this project demonstrates

- End-to-end RAG system design, not just a wrapper around an embeddings API
- Real engineering tradeoffs: parent-child chunking, hybrid retrieval, RRF fusion, explicit hallucination mitigation
- Security conscious design: RBAC enforced at the data layer, JWT auth, password hashing
- Multimodal document understanding: structured table extraction, not just plain text
- Full containerization and reproducible deployment via Docker Compose
- Local-first architecture suitable for regulated/privacy sensitive enterprise contexts
