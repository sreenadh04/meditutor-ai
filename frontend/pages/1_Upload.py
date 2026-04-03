"""
MediTutor AI — Page 1: Upload PDF with User Isolation
"""

import streamlit as st
import requests
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Upload PDF — MediTutor AI", page_icon="📤", layout="wide")

BASE_BACKEND = os.getenv(
    "BACKEND_URL",
    "https://meditutor-backend-v2.onrender.com"
)

API_URL = f"{BASE_BACKEND}/api/v1"


# ─── Helper functions (copied from app.py for consistency) ────────────────────
def get_or_create_user_id() -> str:
    """Get existing user_id or create new one."""
    if "user_id" in st.session_state and st.session_state.user_id:
        return st.session_state.user_id
    
    # Try to load from file
    user_id_file = Path(".user_id")
    if user_id_file.exists():
        try:
            with open(user_id_file, "r") as f:
                user_id = f.read().strip()
                if user_id and len(user_id) >= 32:
                    st.session_state.user_id = user_id
                    return user_id
        except Exception:
            pass
    
    # Generate new
    import uuid
    new_user_id = str(uuid.uuid4())
    st.session_state.user_id = new_user_id
    try:
        with open(user_id_file, "w") as f:
            f.write(new_user_id)
    except Exception:
        pass
    
    return new_user_id


def get_api_headers() -> dict:
    """Get headers for API requests."""
    user_id = get_or_create_user_id()
    return {
        "X-User-ID": user_id,
        "Accept": "application/json",
    }


def get_upload_headers() -> dict:
    """Get headers for file upload (no Content-Type, let requests set it)."""
    user_id = get_or_create_user_id()
    return {
        "X-User-ID": user_id,
        "Accept": "application/json",
    }


# Ensure user_id exists
get_or_create_user_id()

# ─── Shared CSS snippet ──────────────────────────────────────────────────────
st.markdown("""
<style>
.upload-zone { border: 2px dashed #6366f1; border-radius: 12px; padding: 2rem; text-align: center; background: #fafafa; }
.doc-row { display: flex; justify-content: space-between; align-items: center;
           padding: 0.8rem 1rem; border: 1px solid #e2e8f0; border-radius: 8px;
           margin-bottom: 0.5rem; background: white; }
.user-badge { background: #e2e8f0; border-radius: 6px; padding: 0.2rem 0.5rem; font-size: 0.7rem; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("📤 Upload PDF")
st.caption("Upload any textbook or study material. Supported: text-based PDFs (not scanned images).")

# Show user context
user_id = get_or_create_user_id()
st.markdown(f'<span class="user-badge">👤 Uploading as: {user_id[:8]}...{user_id[-4:]}</span>', unsafe_allow_html=True)
st.markdown("---")

# ─── Upload Section ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
    help="Max 50 MB. Must be a text-based PDF (not a scanned image).",
)

if uploaded:
    st.markdown(f"**Selected:** `{uploaded.name}` — {uploaded.size / 1024:.1f} KB")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        upload_clicked = st.button("🚀 Process & Index PDF", type="primary", use_container_width=True)
    
    if upload_clicked:
        with st.spinner("📖 Reading PDF... extracting text... building vector index..."):
            try:
                headers = get_upload_headers()
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                
                # Also send user_id in form data for backward compatibility
                data = {"user_id": user_id}
                
                resp = requests.post(
                    f"{API_URL}/pdf/upload",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=120
                )
                
                if resp.status_code == 200:
                    doc = resp.json()
                    st.success(f"✅ Successfully processed **{doc['filename']}**")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📑 Pages", doc["total_pages"])
                    col2.metric("🧩 Chunks", doc["total_chunks"])
                    col3.metric("🆔 Doc ID", doc["id"][:8] + "...")
                    
                    st.info("👈 Select this document from the sidebar to start studying!")
                    st.session_state["selected_doc_id"] = doc["id"]
                    st.session_state["selected_doc_name"] = doc["filename"]
                    
                elif resp.status_code == 401:
                    st.error("❌ Authentication failed. Please refresh the page and try again.")
                elif resp.status_code == 422:
                    st.error("❌ Could not extract text. This PDF may be image-only (scanned). Try a text-based PDF.")
                elif resp.status_code == 413:
                    st.error("❌ File too large. Maximum size is 50 MB.")
                else:
                    error_detail = resp.json().get('detail', 'Unknown error')
                    st.error(f"❌ Upload failed: {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure the FastAPI server is running.")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The PDF may be too large or complex.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()

# ─── Existing Documents ────────────────────────────────────────────────────────
st.subheader("📚 Your Uploaded Documents")

try:
    headers = get_api_headers()
    resp = requests.get(
        f"{API_URL}/pdf/list",
        headers=headers,
        timeout=10
    )
    
    if resp.status_code == 200:
        docs = resp.json().get("documents", [])
        
        if not docs:
            st.info("No documents uploaded yet. Upload your first PDF above.")
        else:
            for doc in docs:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    with col1:
                        st.markdown(f"**📄 {doc['filename']}**")
                        st.caption(f"ID: {doc['id'][:16]}...")
                    with col2:
                        st.markdown(f"**{doc['total_pages']}** pages")
                    with col3:
                        st.markdown(f"**{doc['total_chunks']}** chunks")
                    with col4:
                        if st.button("Select", key=f"sel_{doc['id']}", use_container_width=True):
                            st.session_state["selected_doc_id"] = doc["id"]
                            st.session_state["selected_doc_name"] = doc["filename"]
                            st.success(f"Selected: {doc['filename']}")
                    with col5:
                        if st.button("🗑️ Delete", key=f"del_{doc['id']}", use_container_width=True):
                            with st.spinner("Deleting..."):
                                del_headers = get_api_headers()
                                del_resp = requests.delete(
                                    f"{API_URL}/pdf/{doc['id']}",
                                    headers=del_headers,
                                    timeout=30
                                )
                                if del_resp.ok:
                                    st.success("Deleted!")
                                    # Clear selection if this was selected
                                    if st.session_state.get("selected_doc_id") == doc["id"]:
                                        st.session_state["selected_doc_id"] = None
                                        st.session_state["selected_doc_name"] = None
                                    st.rerun()
                                else:
                                    st.error(f"Delete failed: {del_resp.status_code}")
                    st.divider()
                    
    elif resp.status_code == 401:
        st.error("❌ Authentication error. Please refresh the page.")
    else:
        st.warning(f"Could not fetch document list (HTTP {resp.status_code})")
        
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to backend. Is it running?")
except Exception as e:
    st.error(f"Error fetching documents: {str(e)}")

# ─── Tips ──────────────────────────────────────────────────────────────────────
with st.expander("💡 Tips for best results"):
    st.markdown("""
    - **Text-based PDFs work best** — PDFs where you can select/copy text
    - **Avoid scanned PDFs** — These contain images of text, not actual text
    - **Large PDFs (200+ pages)** — Take 1-2 minutes to process; please wait
    - **Multiple PDFs** — Upload as many as you need; switch via the sidebar
    - **Re-upload** — If a document seems broken, delete and re-upload it
    - **Your documents are private** — Each user has isolated storage
    """)

# ─── Storage Stats ────────────────────────────────────────────────────────────
with st.expander("📊 Your Storage Usage"):
    try:
        headers = get_api_headers()
        resp = requests.get(
            f"{API_URL}/pdf/stats/summary",
            headers=headers,
            timeout=5
        )
        if resp.status_code == 200:
            stats = resp.json()
            col1, col2 = st.columns(2)
            col1.metric("📄 Documents", stats.get("documents", {}).get("total", 0))
            col2.metric("🧩 Total Chunks", stats.get("storage", {}).get("total_chunks", 0))
            st.caption(f"Approximate storage: ~{stats.get('storage', {}).get('approx_size_kb', 0):.0f} KB")
        else:
            st.info("Login to see your storage usage")
    except Exception:
        st.info("Login to see your storage usage")