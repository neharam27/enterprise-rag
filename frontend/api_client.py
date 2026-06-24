import requests

BASE_URL = "http://localhost:8000"

def login(username: str, password: str) -> dict | None:
    try:
        resp = requests.post(f"{BASE_URL}/login", json={"username": username, "password": password})
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.exceptions.ConnectionError:
        return None

def query(token: str, question: str, top_k: int = 5, use_reranking: bool = True) -> dict | None:
    try:
        resp = requests.post(
            f"{BASE_URL}/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": question, "top_k": top_k, "use_reranking": use_reranking}
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.json().get("detail", "Unknown error")}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the API running on port 8000?"}

def create_user(token: str, username: str, password: str, role: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/admin/create-user",
        headers={"Authorization": f"Bearer {token}"},
        json={"username": username, "password": password, "role": role}
    )
    return resp.json() if resp.status_code == 200 else {"error": resp.json().get("detail")}

def upload_document(token: str, file) -> dict:
    resp = requests.post(
        f"{BASE_URL}/admin/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (file.name, file.getvalue())}
    )
    return resp.json() if resp.status_code == 200 else {"error": resp.json().get("detail")}