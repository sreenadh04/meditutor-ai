"""
MediTutor AI — Page 3: Flashcards with User Isolation
"""

import streamlit as st
import requests
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Flashcards — MediTutor AI", page_icon="🃏", layout="wide")

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
.fc-question {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white; border-radius: 16px; padding: 2.5rem 2rem;
    text-align: center; min-height: 160px; font-size: 1.15rem;
    box-shadow: 0 8px 24px rgba(79,70,229,0.3);
}
.fc-answer {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white; border-radius: 16px; padding: 2.5rem 2rem;
    text-align: center; min-height: 160px; font-size: 1.1rem;
    box-shadow: 0 8px 24px rgba(5,150,105,0.3);
}
.diff-easy   { background:#dcfce7; color:#166534; border-radius:6px; padding:2px 8px; font-size:0.8rem; }
.diff-medium { background:#fef9c3; color:#854d0e; border-radius:6px; padding:2px 8px; font-size:0.8rem; }
.diff-hard   { background:#fee2e2; color:#991b1b; border-radius:6px; padding:2px 8px; font-size:0.8rem; }
.user-badge { background: #e2e8f0; border-radius: 6px; padding: 0.2rem 0.5rem; font-size: 0.7rem; font-family: monospace; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("🃏 Flashcards")
st.caption("Auto-generated study cards — flip to reveal the answer. Export to Anki.")

# Show user context
user_id = get_or_create_user_id()
st.markdown(f'<span class="user-badge">👤 User: {user_id[:8]}...{user_id[-4:]}</span>', unsafe_allow_html=True)
st.markdown("---")

doc_id = st.session_state.get("selected_doc_id")
doc_name = st.session_state.get("selected_doc_name", "")
if not doc_id:
    st.warning("👈 Please select a document first.")
    st.stop()

st.info(f"📄 Document: **{doc_name}**")

# ─── Generation Controls ───────────────────────────────────────────────────────
with st.expander("⚙️ Generation Settings", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input(
            "Topic / Chapter (optional)",
            placeholder="e.g. Cardiac Pharmacology, Chapter 3",
            help="Leave blank to generate from the whole document."
        )
    with col2:
        count = st.slider("Number of flashcards", 5, 30, 10)

    gen_btn = st.button("✨ Generate Flashcards", type="primary", use_container_width=True)

if gen_btn:
    with st.spinner("🤖 Generating flashcards... (may take 15-30s on free tier)"):
        try:
            headers = get_api_headers()
            payload = {
                "document_id": doc_id,
                "count": count,
                "topic": topic.strip() if topic.strip() else None,
            }
            resp = requests.post(
                f"{API_URL}/flashcards/generate",
                json=payload,
                headers=headers,
                timeout=120
            )

            if resp.status_code == 200:
                data = resp.json()
                st.session_state["current_flashcards"] = data["flashcards"]
                st.session_state["flashcard_index"] = 0
                st.session_state["show_answer"] = False
                model = data.get("model_used", "")
                cached = data.get("cached", False)
                st.success(f"✅ Generated {data['total_generated']} flashcards — Model: `{model}`{'  ⚡ (cached)' if cached else ''}")
            elif resp.status_code == 401:
                st.error("❌ Authentication error. Please refresh the page.")
            else:
                st.error(f"❌ {resp.json().get('detail', 'Generation failed')}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Timed out. Try reducing the count or try again.")
        except Exception as e:
            st.error(f"❌ {e}")

# ─── Flashcard Viewer ──────────────────────────────────────────────────────────
cards = st.session_state.get("current_flashcards", [])

if cards:
    st.divider()
    idx = st.session_state.get("flashcard_index", 0)
    idx = max(0, min(idx, len(cards) - 1))
    card = cards[idx]

    # Progress bar
    progress_pct = (idx + 1) / len(cards)

    difficulty = card.get("difficulty", "medium")
    topic_text = card.get("topic", "General")

    st.markdown(
        f"**Card {idx+1} of {len(cards)}** — "
        f'<span class="diff-{difficulty}">{difficulty}</span>'
        + " &nbsp; 🏷️ " + topic_text,
        unsafe_allow_html=True,
    )
    st.progress(progress_pct)
    st.markdown("<br>", unsafe_allow_html=True)

    # Card display
    show = st.session_state.get("show_answer", False)

    if not show:
        st.markdown(
            f'<div class="fc-question">❓<br><br><b>{card["question"]}</b></div>',
            unsafe_allow_html=True,
        )
        if st.button("👁️ Reveal Answer", use_container_width=True):
            st.session_state["show_answer"] = True
            st.rerun()
    else:
        st.markdown(
            f'<div class="fc-answer">✅<br><br>{card["answer"]}</div>',
            unsafe_allow_html=True,
        )
        
        # Record review (optional feature)
        col_rev1, col_rev2, col_rev3 = st.columns(3)
        with col_rev1:
            if st.button("😊 Easy", use_container_width=True):
                st.toast("Great! You knew this one.")
                # Could call API to record review
        with col_rev2:
            if st.button("🤔 Medium", use_container_width=True):
                st.toast("Keep practicing!")
        with col_rev3:
            if st.button("😓 Hard", use_container_width=True):
                st.toast("Review this again soon.")
        
        if st.button("🔁 Hide Answer", use_container_width=True):
            st.session_state["show_answer"] = False
            st.rerun()

    # Navigation
    st.markdown("<br>", unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅️ Previous", disabled=(idx == 0), use_container_width=True):
            st.session_state["flashcard_index"] = idx - 1
            st.session_state["show_answer"] = False
            st.rerun()
    with nav2:
        jump = st.number_input("Jump to card #", 1, len(cards), idx + 1, label_visibility="collapsed")
        if st.button("Jump", use_container_width=True):
            st.session_state["flashcard_index"] = int(jump) - 1
            st.session_state["show_answer"] = False
            st.rerun()
    with nav3:
        if st.button("Next ➡️", disabled=(idx == len(cards) - 1), use_container_width=True):
            st.session_state["flashcard_index"] = idx + 1
            st.session_state["show_answer"] = False
            st.rerun()

    # ─── All Cards Table ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 View All Flashcards"):
        for i, c in enumerate(cards):
            with st.container():
                q_col, a_col = st.columns([1, 1])
                with q_col:
                    st.markdown(f"**Q{i+1}:** {c['question']}")
                with a_col:
                    st.markdown(f"**A:** {c['answer']}")
                st.caption(f"Topic: {c.get('topic','—')} | Difficulty: {c.get('difficulty','medium')}")
                st.divider()

    # ─── Export Button with Auth Headers ────────────────────────────────────────
    st.subheader("📥 Export to Anki")
    try:
        headers = get_api_headers()
        csv_resp = requests.get(
            f"{API_URL}/flashcards/export/{doc_id}",
            headers=headers,
            timeout=10
        )
        if csv_resp.status_code == 200:
            st.download_button(
                "⬇️ Download Anki CSV",
                data=csv_resp.content,
                file_name=f"flashcards_{doc_id[:8]}_{user_id[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        elif csv_resp.status_code == 401:
            st.info("Please refresh the page to export flashcards.")
        else:
            st.info("Generate flashcards first, then export.")
    except Exception as e:
        st.warning(f"Export unavailable: {str(e)[:50]}")