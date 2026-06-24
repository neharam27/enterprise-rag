import streamlit as st
from api_client import login, query, create_user, upload_document

st.set_page_config(page_title="Enterprise Document Intelligence", layout="wide")

# ---------- SESSION STATE ----------
if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.chat_history = []

# ---------- LOGIN SCREEN ----------
def login_screen():
    st.title("🏢 Enterprise Document Intelligence Platform")
    st.caption("Hybrid RAG · RBAC · Multimodal retrieval — running fully local on Ollama")

    col1, col2 = st.columns([1, 2])
    with col1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

            if submitted:
                result = login(username, password)
                if result:
                    st.session_state.token = result["access_token"]
                    st.session_state.role = result["role"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials or backend unreachable.")

        st.divider()
        st.caption("**Demo accounts:**")
        st.code("admin1 / adminpass123\nanalyst1 / analystpass123\nviewer1 / viewerpass123", language=None)

# ---------- SIDEBAR ----------
def sidebar():
    with st.sidebar:
        st.subheader(f"👤 {st.session_state.username}")
        st.caption(f"Role: `{st.session_state.role}`")

        if st.button("Log out", use_container_width=True):
            st.session_state.token = None
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()

        st.divider()

        with st.expander("⚙️ Query settings"):
            st.session_state.top_k = st.slider("Top-K results", 1, 10, 5)
            st.session_state.use_reranking = st.checkbox("Use LLM reranking", value=True)

# ---------- CHAT INTERFACE ----------
def chat_interface():
    st.title("💬 Ask your documents")

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            render_answer(entry["response"])

    question = st.chat_input("Ask a question about the ingested documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Running hybrid search → RRF → rerank → generation..."):
                response = query(
                    st.session_state.token,
                    question,
                    top_k=st.session_state.get("top_k", 5),
                    use_reranking=st.session_state.get("use_reranking", True)
                )
            render_answer(response)

        st.session_state.chat_history.append({"question": question, "response": response})

def render_answer(response: dict):
    if not response or "error" in response:
        st.error(response.get("error", "Something went wrong.") if response else "No response.")
        return

    st.write(response["answer"])

    confidence_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    grounded_text = "✅ Grounded in retrieved context" if response["grounded"] else "⚠️ Not grounded — model could not confirm an answer in context"

    st.caption(f"{confidence_colors.get(response['confidence'], '⚪')} Confidence: **{response['confidence']}** · {grounded_text}")

    if response["sources"]:
        with st.expander(f"📚 View {len(response['sources'])} cited source(s)"):
            for src in response["sources"]:
                st.markdown(f"**Source {src['source_id']}** — `{src['source_file']}`, page {src['page_number']} · type: `{src['chunk_type']}` · relevance: `{src['relevance_score']}`")
                st.code(src["content"], language=None)

# ---------- ADMIN PANEL ----------
def admin_panel():
    st.title("🛠️ Admin Panel")

    tab1, tab2 = st.tabs(["📄 Upload Document", "👥 Create User"])

    with tab1:
        st.write("Upload a PDF to be ingested. **Note:** uploading only stores the file — run the ingestion script separately to index it.")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        if uploaded_file and st.button("Upload"):
            result = upload_document(st.session_state.token, uploaded_file)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(result["message"])
                st.code(f"python ingest.py {result['path']}", language="bash")

    with tab2:
        with st.form("create_user_form"):
            new_username = st.text_input("New username")
            new_password = st.text_input("New password", type="password")
            new_role = st.selectbox("Role", ["admin", "analyst", "viewer"])
            if st.form_submit_button("Create user"):
                result = create_user(st.session_state.token, new_username, new_password, new_role)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(result["message"])

# ---------- MAIN ROUTER ----------
if not st.session_state.token:
    login_screen()
else:
    sidebar()

    if st.session_state.role == "admin":
        tab1, tab2 = st.tabs(["💬 Chat", "🛠️ Admin"])
        with tab1:
            chat_interface()
        with tab2:
            admin_panel()
    else:
        chat_interface()