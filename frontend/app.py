"""
MediTutor AI — Streamlit Frontend (Production Ready)
Main entry point with user isolation and persistent user_id.
"""

import streamlit as st
import os
import uuid
import requests
import json
from pathlib import Path

# ── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="MediTutor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_BACKEND = os.getenv(
    "BACKEND_URL",
    "https://meditutor-backend-v2.onrender.com"
)
API_URL = f"{BASE_BACKEND}/api/v1"

# ─── User ID Management (Persistent) ─────────────────────────────────────────
def get_or_create_user_id() -> str:
    """
    Get existing user_id from localStorage or create new one.
    Persists across browser sessions using Streamlit's secrets.
    """
    # Try to get from session state first
    if "user_id" in st.session_state and st.session_state.user_id:
        return st.session_state.user_id
    
    # Try to load from file (persists across server restarts)
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
    
    # Generate new user_id
    new_user_id = str(uuid.uuid4())
    st.session_state.user_id = new_user_id
    
    # Save to file for persistence
    try:
        with open(user_id_file, "w") as f:
            f.write(new_user_id)
    except Exception:
        pass
    
    return new_user_id


def get_api_headers() -> dict:
    """
    Get headers for API requests including user authentication.
    Call this for every backend request.
    """
    user_id = get_or_create_user_id()
    return {
        "X-User-ID": user_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def get_upload_headers() -> dict:
    """
    Get headers for file upload requests (no Content-Type).
    """
    user_id = get_or_create_user_id()
    return {
        "X-User-ID": user_id,
        "Accept": "application/json",
    }


# ─── Session State Defaults ──────────────────────────────────────────────────
defaults = {
    "api_url": API_URL,
    "selected_doc_id": None,
    "selected_doc_name": None,
    "session_id": None,
    "chat_history": [],
    "current_mcqs": [],
    "current_flashcards": [],
    "mcq_answers": {},
    "quiz_submitted": False,
    "flashcard_index": 0,
    "show_answer": False,
}

# Initialize user_id (persistent)
user_id = get_or_create_user_id()

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font and background */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
    font-size: 0.82rem;
}

/* Cards */
.mt-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.mt-card-blue {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.mt-card-green {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.mt-card-red {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1px solid #fecdd3;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.mt-card-yellow {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid #fde68a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

/* Source citation box */
.source-box {
    background: #f8fafc;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #475569;
}

/* Flashcard */
.flashcard {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.35);
}
.flashcard-answer {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 8px 24px rgba(17, 153, 142, 0.35);
}

/* MCQ option */
.option-btn {
    width: 100%;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    border: 2px solid #e2e8f0;
    background: white;
    text-align: left;
    cursor: pointer;
    margin-bottom: 0.5rem;
    transition: all 0.2s;
}
.option-correct { border-color: #22c55e !important; background: #f0fdf4 !important; }
.option-wrong { border-color: #ef4444 !important; background: #fff1f2 !important; }

/* Metric chips */
.metric-chip {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
}
.chip-blue { background: #dbeafe; color: #1d4ed8; }
.chip-green { background: #dcfce7; color: #15803d; }
.chip-red { background: #fee2e2; color: #b91c1c; }
.chip-yellow { background: #fef3c7; color: #b45309; }

/* Progress bar custom */
.prog-bar-container {
    background: #e2e8f0;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 0.3rem 0;
}
.prog-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #312e81 50%, #0f172a 100%);
    color: white;
    padding: 2rem 2rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2rem; }
.hero p  { margin: 0.4rem 0 0; color: #a5b4fc; font-size: 1.05rem; }

/* User badge */
.user-badge {
    background: #334155;
    border-radius: 8px;
    padding: 0.3rem 0.6rem;
    font-size: 0.7rem;
    font-family: monospace;
    color: #94a3b8;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar: Document Selector ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MediTutor AI")
    
    # Show user badge (for debugging/awareness)
    st.markdown(
        f'<div class="user-badge">👤 User: {user_id[:8]}...{user_id[-4:]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Fetch documents with auth headers
    try:
        headers = get_api_headers()
        resp = requests.get(
            f"{st.session_state.api_url}/pdf/list",
            headers=headers,
            timeout=5
        )
        if resp.ok:
            docs = resp.json().get("documents", [])
        else:
            docs = []
            st.warning(f"Backend error: {resp.status_code}")
    except Exception as e:
        docs = []
        st.warning(f"⚠️ Backend offline: {str(e)[:50]}")

    if docs:
        doc_options = {f"📄 {d['filename'][:30]}": d["id"] for d in docs}
        selected_label = st.selectbox(
            "Active Document",
            list(doc_options.keys()),
            index=0,
        )
        st.session_state.selected_doc_id = doc_options[selected_label]
        st.session_state.selected_doc_name = selected_label.replace("📄 ", "")

        # Doc metadata
        sel_doc = next((d for d in docs if d["id"] == st.session_state.selected_doc_id), None)
        if sel_doc:
            st.markdown(f"""
            <div style='font-size:0.8rem; color:#94a3b8; margin-top:0.5rem;'>
            📑 {sel_doc['total_pages']} pages &nbsp;|&nbsp; 🧩 {sel_doc['total_chunks']} chunks
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Upload a PDF to get started.")
        st.session_state.selected_doc_id = None

    st.markdown("---")
    st.markdown("**Navigate**")
    st.page_link("app.py",              label="🏠 Home")
    st.page_link("pages/1_Upload.py",   label="📤 Upload PDF")
    st.page_link("pages/2_QA_Chat.py",  label="💬 Ask Questions")
    st.page_link("pages/3_Flashcards.py", label="🃏 Flashcards")
    st.page_link("pages/4_MCQ_Quiz.py", label="📝 MCQ Quiz")
    st.page_link("pages/5_Progress.py", label="📊 Progress")
    st.page_link("pages/6_Prereq.py",   label="🔍 Prerequisites")

    st.markdown("---")
    
    # Optional: Clear user data button (GDPR)
    if st.button("🗑️ Clear My Data", type="secondary", use_container_width=True):
        try:
            headers = get_api_headers()
            resp = requests.delete(
                f"{st.session_state.api_url.replace('/api/v1', '')}/api/v1/user/data",
                headers=headers,
                timeout=30
            )
            if resp.ok:
                st.success("Your data has been deleted. Refresh to start fresh.")
                # Clear local user_id file
                user_id_file = Path(".user_id")
                if user_id_file.exists():
                    user_id_file.unlink()
                st.session_state.clear()
                st.rerun()
            else:
                st.error(f"Failed to delete data: {resp.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.markdown(
        "<div style='font-size:0.75rem;color:#64748b;text-align:center;margin-top:1rem;'>"
        "MediTutor AI v2.0 • User Isolated</div>",
        unsafe_allow_html=True,
    )

# ─── Start Study Session (if document selected and no session) ─────────────────
if st.session_state.selected_doc_id and not st.session_state.get("session_id"):
    try:
        headers = get_api_headers()
        payload = {
            "document_id": st.session_state.selected_doc_id,
            # No student_id needed — backend uses X-User-ID header
        }
        resp = requests.post(
            f"{API_URL}/progress/session/start",
            json=payload,
            headers=headers,
            timeout=5,
        )
        if resp.ok:
            st.session_state["session_id"] = resp.json()["session_id"]
        else:
            st.warning("Could not start study session. Progress may not be tracked.")
    except Exception as e:
        st.warning(f"Session error: {str(e)[:50]}")

# ─── Home Page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 MediTutor AI</h1>
  <p>Your AI-powered study assistant — Upload any textbook and start learning smarter.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="mt-card-blue">
      <h3>📤 Upload PDF</h3>
      <p>Upload any medical or academic textbook. AI extracts, chunks, and indexes it instantly.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="mt-card-green">
      <h3>💬 Ask Questions</h3>
      <p>RAG-powered Q&A with source citations — always grounded in your textbook content.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="mt-card-yellow">
      <h3>🃏 Flashcards + MCQs</h3>
      <p>Auto-generate study cards and quizzes. Export to Anki. Track your progress over time.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 🚀 Quick Start")
st.markdown("""
1. Click **📤 Upload PDF** in the sidebar and upload your textbook  
2. Go to **💬 Ask Questions** to get AI answers with citations  
3. Generate **🃏 Flashcards** or a **📝 MCQ Quiz** from the content  
4. Check **📊 Progress** to see your weak topics  
5. Use **🔍 Prerequisites** before tackling complex topics  
""")

# API status
st.markdown("### 🔌 System Status")
c1, c2 = st.columns(2)
with c1:
    try:
        headers = get_api_headers()
        r = requests.get(
            f"{st.session_state.api_url.replace('/api/v1', '')}/health",
            headers=headers,
            timeout=4
        )
        if r.ok:
            data = r.json()
            st.success("✅ Backend connected")
            models = data.get("llm", {})
            groq_ok = models.get("groq", {}).get("configured", False)
            hf_ok = models.get("huggingface", {}).get("configured", False)
            st.markdown(f"**Groq API:** {'✅' if groq_ok else '❌ Not configured'}")
            st.markdown(f"**HuggingFace API:** {'✅' if hf_ok else '❌ Not configured'}")
            
            # Show auth status
            auth = data.get("auth", {})
            st.markdown(f"**Auth Mode:** {auth.get('mode', 'unknown')}")
        else:
            st.error("❌ Backend returned error")
    except Exception as e:
        st.error(f"❌ Backend not reachable — is it running? ({str(e)[:50]})")
with c2:
    st.info(f"**API URL:** `{st.session_state.api_url}`\n\n**User ID:** `{user_id[:8]}...{user_id[-4:]}`")