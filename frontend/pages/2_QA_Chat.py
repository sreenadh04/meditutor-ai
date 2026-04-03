"""
MediTutor AI — Page 2: Q&A Chat with RAG and User Isolation
"""

import streamlit as st
import requests
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Q&A — MediTutor AI", page_icon="💬", layout="wide")

BASE_BACKEND = os.getenv(
    "BACKEND_URL",
    "https://meditutor-backend-v2.onrender.com"
)

API_URL = f"{BASE_BACKEND}/api/v1"


# ─── Helper functions ─────────────────────────────────────────────────────────
def get_or_create_user_id() -> str:
    """Get existing user_id or create new one."""
    if "user_id" in st.session_state and st.session_state.user_id:
        return st.session_state.user_id
    
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
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# Ensure user_id exists
get_or_create_user_id()

st.markdown("""
<style>
.chat-user { background: #eff6ff; border-radius: 12px 12px 2px 12px; padding: 0.9rem 1.2rem; margin: 0.5rem 0; }
.chat-ai   { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px 12px 12px 2px; padding: 0.9rem 1.2rem; margin: 0.5rem 0; }
.source-pill { display: inline-block; background: #ede9fe; color: #5b21b6; border-radius: 6px; padding: 0.25rem 0.6rem; font-size: 0.78rem; margin: 0.2rem; }
.source-text { background: #fafafa; border-left: 3px solid #818cf8; padding: 0.6rem 0.9rem; border-radius: 0 8px 8px 0; font-size: 0.83rem; color: #475569; margin: 0.3rem 0; }
.user-badge { background: #e2e8f0; border-radius: 6px; padding: 0.2rem 0.5rem; font-size: 0.7rem; font-family: monospace; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("💬 Ask Questions")
st.caption("RAG-powered Q&A — every answer is grounded in your textbook with source citations.")

# Show user context
user_id = get_or_create_user_id()
st.markdown(f'<span class="user-badge">👤 User: {user_id[:8]}...{user_id[-4:]}</span>', unsafe_allow_html=True)
st.markdown("---")

# Guard: need a document selected
doc_id = st.session_state.get("selected_doc_id")
doc_name = st.session_state.get("selected_doc_name", "Unknown")
if not doc_id:
    st.warning("👈 Please upload and select a document from the sidebar first.")
    st.stop()

st.info(f"📄 Active document: **{doc_name}**")

# ─── Session Management (with headers) ────────────────────────────────────────
if not st.session_state.get("session_id"):
    try:
        headers = get_api_headers()
        # Note: No student_id in body — backend uses X-User-ID header
        payload = {"document_id": doc_id}
        r = requests.post(
            f"{API_URL}/progress/session/start",
            json=payload,
            headers=headers,
            timeout=5,
        )
        if r.status_code == 200:
            st.session_state["session_id"] = r.json()["session_id"]
        elif r.status_code == 401:
            st.error("❌ Authentication error. Please refresh the page.")
        else:
            st.warning("Could not start study session. Progress may not be tracked.")
    except Exception as e:
        st.warning(f"Session error: {str(e)[:50]}")

# ─── Chat History Display ──────────────────────────────────────────────────────
chat_container = st.container()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

with chat_container:
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑‍🎓 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🧠 <b>MediTutor AI:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            
            # Show sources if available
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} Source(s) used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        page_label = f"Page {src['page_number']}" if src.get("page_number") else "Unknown page"
                        score = src.get("relevance_score", 0)
                        st.markdown(
                            f'<span class="source-pill">Source {i} • {page_label} • score: {score:.2f}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="source-text">{src["text"]}</div>',
                            unsafe_allow_html=True,
                        )
            
            if msg.get("model_used"):
                st.caption(f"🤖 Model: `{msg['model_used']}`")

# ─── Input Area ────────────────────────────────────────────────────────────────
st.divider()

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Ask a question about your textbook...",
        placeholder="e.g. What is the mechanism of action of beta blockers?",
        key="qa_input",
        label_visibility="collapsed",
    )
with col2:
    ask_btn = st.button("Ask 🔍", type="primary", use_container_width=True)

# ─── Example Questions ─────────────────────────────────────────────────────────
with st.expander("💡 Example questions"):
    examples = [
        "What is the main topic of Chapter 1?",
        "Explain the key concepts in this material.",
        "What are the most important definitions?",
        "Summarize the treatment options discussed.",
        "What are the contraindications mentioned?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            question = ex
            ask_btn = True

# ─── Process Question ──────────────────────────────────────────────────────────
if ask_btn and question and question.strip():
    st.session_state["chat_history"].append({"role": "user", "content": question})

    with st.spinner("🔍 Searching textbook and generating answer..."):
        try:
            headers = get_api_headers()
            payload = {
                "document_id": doc_id,
                "question": question,
                "session_id": st.session_state.get("session_id"),
            }
            resp = requests.post(
                f"{API_URL}/qa/ask",
                json=payload,
                headers=headers,
                timeout=90
            )

            if resp.status_code == 200:
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                model_used = data.get("model_used", "unknown")
                cached = data.get("cached", False)

                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "model_used": model_used + (" ⚡cached" if cached else ""),
                })
                st.rerun()

            elif resp.status_code == 401:
                st.error("❌ Authentication error. Please refresh the page.")
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": "Authentication failed. Please refresh the page and try again.",
                    "sources": [],
                    "model_used": "error",
                })
                st.rerun()
            elif resp.status_code == 404:
                st.error("❌ Document index not found. Please re-upload the PDF.")
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": "The document index was not found. Please re-upload your PDF.",
                    "sources": [],
                    "model_used": "error",
                })
                st.rerun()
            elif resp.status_code == 503:
                st.error("❌ AI models unavailable. Check API keys in your .env file.")
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": "The AI service is currently unavailable. Please try again later.",
                    "sources": [],
                    "model_used": "error",
                })
                st.rerun()
            else:
                error_detail = resp.json().get('detail', 'Unknown error')
                st.error(f"❌ Error: {error_detail}")

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The free AI model may be slow — please try again.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Is the FastAPI server running?")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ─── Clear Chat ────────────────────────────────────────────────────────────────
if st.session_state["chat_history"]:
    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()

# ─── Debug Info (only in development) ──────────────────────────────────────────
if os.getenv("DEBUG", "false").lower() == "true":
    with st.expander("🔧 Debug Info"):
        st.json({
            "user_id": user_id[:8] + "...",
            "doc_id": doc_id[:8] + "...",
            "session_id": st.session_state.get("session_id", "None")[:8] + "..." if st.session_state.get("session_id") else "None",
            "api_url": API_URL,
            "chat_history_length": len(st.session_state.get("chat_history", [])),
        })