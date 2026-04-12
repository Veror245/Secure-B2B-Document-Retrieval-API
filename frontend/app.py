import streamlit as st
import requests
import time

# --- Configuration ---
API_URL = "http://rag-backend:8000"

# --- Styling (Modern & Sleek) ---
st.set_page_config(page_title="Secure B2B RAG", page_icon="🔒", layout="wide")

# Custom CSS to hide default Streamlit top margin and style elements
st.markdown("""
<style>

/* === GLOBAL APP STYLING === */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}

/* Remove default padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Hide sidebar nav */
div[data-testid="stSidebarNav"] {
    display: none;
}

/* === AUTH CARD STYLE === */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.04);
    padding: 2rem;
    border-radius: 16px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* === INPUT FIELDS === */
input, textarea {
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 10px !important;
}

input:focus, textarea:focus {
    border: 1px solid #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6;
}

/* === BUTTONS === */
button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    border: none;
    border-radius: 10px;
    transition: all 0.25s ease;
    font-weight: 500;
}

button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.4);
}

/* Sidebar buttons */
section[data-testid="stSidebar"] button {
    border-radius: 8px;
}

/* === CHAT UI === */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 12px;
}

/* User message */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
}

/* Assistant message */
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
}

/* Chat input box */
.stChatFloatingInputContainer {
    background: rgba(15,23,42,0.8);
    border-top: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding-bottom: 20px;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background: rgba(2,6,23,0.9);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 10px;
    border: 1px dashed rgba(255,255,255,0.1);
}

/* === EXPANDERS (Sources) === */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* === TEXT === */
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.5px;
}

/* === TOAST === */
[data-testid="stToast"] {
    border-radius: 10px;
}

/* === SCROLLBAR === */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.15);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.3);
}

</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "token" not in st.session_state:
    st.session_state.token = None
if "view" not in st.session_state:
    st.session_state.view = "login"
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---
def api_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def logout():
    st.session_state.token = None
    st.session_state.messages = []
    st.session_state.view = "login"

# --- VIEWS ---

def auth_view():
    st.title("🔒 Secure Enterprise RAG")
    st.markdown("Login to access your isolated document workspace.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tabs = st.tabs(["Login", "Sign Up"])
        
        # Login Tab
        with tabs[0]:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted:
                    with st.spinner("Authenticating..."):
                        # FastAPI's OAuth2PasswordRequestForm expects 'username' and 'password' in form data
                        res = requests.post(f"{API_URL}/auth/login", data={"username": email, "password": password})
                        if res.status_code == 200:
                            st.session_state.token = res.json().get("access_token")
                            st.rerun()
                        else:
                            st.error(res.json().get("detail", "Login failed."))
                            
        # Register Tab
        with tabs[1]:
            with st.form("register_form"):
                reg_email = st.text_input("Email")
                reg_password = st.text_input("Password", type="password")
                reg_submitted = st.form_submit_button("Create Account", use_container_width=True)
                
                if reg_submitted:
                    with st.spinner("Creating account..."):
                        res = requests.post(f"{API_URL}/auth/register", json={"email": reg_email, "password": reg_password})
                        if res.status_code == 201:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error(res.json().get("detail", "Registration failed."))

def main_app_view():
    # --- SIDEBAR: Document Ingestion & Management ---
    with st.sidebar:
        st.header("📂 My Workspace")
        
        # 1. Upload Section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")
        
        if uploaded_file and st.button("Process Document", use_container_width=True):
            with st.spinner(f"Ingesting & Chunking {uploaded_file.name}..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post(f"{API_URL}/documents/upload", files=files, headers=api_headers())
                
                if res.status_code == 200:
                    data = res.json()
                    st.toast(f"✅ Ingested {data['chunks_created']} chunks. BM25 indexing in background.")
                else:
                    st.error(res.json().get("detail", "Upload failed."))
        
        st.divider()
        
        # 2. Document List Section
        st.subheader("Ingested Files")
        try:
            res = requests.get(f"{API_URL}/documents/my-files", headers=api_headers())
            if res.status_code == 200:
                docs = res.json().get("documents", [])
                if not docs:
                    st.info("No documents uploaded yet.")
                else:
                    for doc in docs:
                        status_icon = "🟢" if doc['status'] == "processed" else "🟡"
                        st.markdown(f"{status_icon} **{doc['filename']}**")
            else:
                st.error("Failed to load documents.")
        except Exception:
            st.error("API Connection Error.")
            
        st.divider()
        st.button("Logout", on_click=logout, use_container_width=True)

    # --- MAIN AREA: Chat / Q&A Interface ---
    st.title("Enterprise RAG Assistant")
    st.markdown("Ask questions against your proprietary data. The model strictly cites your uploaded files.")
    
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Cited Sources"):
                    for i, source in enumerate(msg["sources"]):
                        st.markdown(f"**Source {i+1}: {source.get('file')}** (Page {source.get('page')})")
                        st.code(source.get("preview"), language=None)

    # Chat Input
    if prompt := st.chat_input("E.g., What was the Q3 revenue in the report?"):
        # 1. Append User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call API & Display Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Searching proprietary database..."):
                payload = {"query": prompt} # tenant_id is safely inferred from the JWT Token in the backend!
                try:
                    res = requests.post(f"{API_URL}/query/", json=payload, headers=api_headers())
                    
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer_markdown"]
                        sources = data.get("sources", [])
                        
                        # Display markdown answer
                        st.markdown(answer)
                        
                        # Display sources inside an expander
                        if sources:
                            with st.expander("View Cited Sources"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}: {source.get('file')}** (Page {source.get('page')})")
                                    st.code(source.get("preview"), language=None)
                                    
                        # Append to state
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    else:
                        error_msg = res.json().get("detail", "An error occurred.")
                        st.error(error_msg)
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot connect to the FastAPI backend. Is it running?")

# --- Routing ---
if st.session_state.token is None:
    auth_view()
else:
    main_app_view()