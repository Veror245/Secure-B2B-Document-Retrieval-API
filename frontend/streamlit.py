import streamlit as st
import requests

# ---------------- CONFIG ----------------
API_URL = "http://localhost:8000"  # FastAPI base URL
UPLOAD_URL = f"{API_URL}/documents/upload"
QUERY_URL = f"{API_URL}/query"  # original endpoint

st.set_page_config(page_title="RAG App", layout="wide")

# ---------------- SESSION STATE ----------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Data Ingestion")

TENANT_ID = st.sidebar.text_input("Tenant ID", value="test_user_1")

uploaded_files = st.sidebar.file_uploader(
    "Upload your documents",
    accept_multiple_files=True,
    type=["pdf", "txt", "md"]
)

# Upload handling
if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            with st.sidebar.spinner(f"Uploading {file.name}..."):
                try:
                    response = requests.post(
                        UPLOAD_URL,
                        files={"file": (file.name, file.getvalue())},
                        data={"tenant_id": TENANT_ID}
                    )

                    if response.status_code == 200:
                        st.session_state.uploaded_files.append(file.name)
                        st.sidebar.success(f"✅ {file.name} uploaded")
                    else:
                        st.sidebar.error(f"❌ Failed: {file.name}")

                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")

# Show uploaded files
st.sidebar.subheader("📁 Uploaded Files")
if st.session_state.uploaded_files:
    for f in st.session_state.uploaded_files:
        st.sidebar.markdown(f"- {f}")
else:
    st.sidebar.info("No files uploaded yet.")

# ---------------- MAIN UI ----------------
st.title("⚡ RAG Assistant")
st.markdown(f"**Tenant ID:** `{TENANT_ID}`")

query = st.text_input("💬 Enter your question")

# ---------------- QUERY ----------------
if st.button("Ask"):
    if not query:
        st.warning("Please enter a question")
    else:
        with st.spinner("🔍 Retrieving and generating answer..."):
            try:
                response = requests.post(
                    QUERY_URL,
                    json={
                        "query": query,
                        "tenant_id": TENANT_ID
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    # Show answer immediately
                    st.markdown("## 📖 Answer")
                    st.markdown(data.get("answer_markdown", ""), unsafe_allow_html=True)

                    if not data.get("is_relevant", True):
                        st.warning("⚠️ The retrieved context may not be fully relevant.")

                    sources = data.get("sources", [])
                    if sources:
                        st.markdown("## 📚 Sources")
                        for src in sources:
                            st.markdown(f"- {src}")

                else:
                    st.error(f"Error: {response.text}")

            except Exception as e:
                st.error(f"Request failed: {str(e)}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Streamlit + FastAPI RAG")