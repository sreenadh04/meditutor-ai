"""
MediTutor AI — Page 5: Progress Dashboard with User Isolation
"""

import streamlit as st
import requests
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Progress — MediTutor AI", page_icon="📊", layout="wide")

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
        "Accept": "application/json",
    }


# Ensure user_id exists
get_or_create_user_id()

st.title("📊 Progress Dashboard")
st.caption("Track your performance, spot weak areas, and measure improvement over time.")

# Show user context
user_id = get_or_create_user_id()
st.markdown(f"👤 **Your progress data** — User ID: `{user_id[:8]}...{user_id[-4:]}`")
st.markdown("---")

doc_id = st.session_state.get("selected_doc_id")
doc_name = st.session_state.get("selected_doc_name", "")
if not doc_id:
    st.warning("👈 Select a document from the sidebar to view your progress.")
    st.stop()

st.info(f"📄 Document: **{doc_name}**")

# ─── Fetch Progress with User Headers ─────────────────────────────────────────
try:
    headers = get_api_headers()
    # No student_id param needed — backend uses X-User-ID header
    resp = requests.get(
        f"{API_URL}/progress/{doc_id}",
        headers=headers,
        timeout=10,
    )
    
    if resp.status_code == 200:
        prog = resp.json()
    elif resp.status_code == 401:
        st.error("❌ Authentication error. Please refresh the page.")
        st.stop()
    elif resp.status_code == 404:
        st.error("Document not found.")
        st.stop()
    else:
        st.error(f"Failed to load progress: {resp.text[:200]}")
        st.stop()
        
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to backend. Is it running?")
    st.stop()
except Exception as e:
    st.error(f"Backend error: {e}")
    st.stop()

# ─── Overview Metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📝 Total Attempts",  prog.get("total_attempts", 0))
c2.metric("✅ Correct",         prog.get("total_correct", 0))
c3.metric("🎯 Accuracy",        f"{prog.get('overall_accuracy', 0):.1f}%")
c4.metric("⚠️ Weak Topics",     len(prog.get("weak_topics", [])))

# ─── Accuracy Bar ─────────────────────────────────────────────────────────────
acc = prog.get("overall_accuracy", 0)
bar_color = "#22c55e" if acc >= 70 else "#f59e0b" if acc >= 50 else "#ef4444"
st.markdown(f"""
<div style="margin: 1rem 0;">
  <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#64748b;">
    <span>Overall Accuracy</span><span>{acc:.1f}%</span>
  </div>
  <div style="background:#e2e8f0; border-radius:999px; height:14px; overflow:hidden; margin-top:4px;">
    <div style="width:{acc}%; height:100%; background:{bar_color}; border-radius:999px; transition:width 0.5s;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Weak & Strong Topics ──────────────────────────────────────────────────────
col_weak, col_strong = st.columns(2)

with col_weak:
    st.subheader("⚠️ Weak Topics (< 60%)")
    weak_topics = prog.get("weak_topics", [])
    if weak_topics:
        for t in weak_topics:
            st.markdown(
                f'<span style="display:inline-block;background:#fee2e2;color:#b91c1c;'
                f'border-radius:6px;padding:4px 10px;margin:3px;font-size:0.85rem;">🔴 {t}</span>',
                unsafe_allow_html=True,
            )
        st.caption("💡 Focus your revision on these topics!")
    else:
        st.success("No weak topics yet — keep practicing!")

with col_strong:
    st.subheader("🏆 Strong Topics (≥ 80%)")
    strong_topics = prog.get("strong_topics", [])
    if strong_topics:
        for t in strong_topics:
            st.markdown(
                f'<span style="display:inline-block;background:#dcfce7;color:#166534;'
                f'border-radius:6px;padding:4px 10px;margin:3px;font-size:0.85rem;">🟢 {t}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Keep practicing to build strong topics!")

st.divider()

# ─── Topic Breakdown Table ─────────────────────────────────────────────────────
st.subheader("📋 Topic-by-Topic Breakdown")

topics = prog.get("topics", [])
if topics:
    # Sort by accuracy ascending (weakest first)
    topics_sorted = sorted(topics, key=lambda x: x["accuracy"])
    
    for t in topics_sorted:
        accuracy = t["accuracy"]
        bar_c = "#22c55e" if accuracy >= 70 else "#f59e0b" if accuracy >= 50 else "#ef4444"
        icon = "🔴" if t["is_weak"] else "🟡" if accuracy < 80 else "🟢"
        
        with st.container():
            tc1, tc2, tc3 = st.columns([3, 1, 2])
            with tc1:
                st.markdown(f"{icon} **{t['topic']}**")
            with tc2:
                st.markdown(f"**{t['correct']}/{t['attempts']}**")
            with tc3:
                st.markdown(f"""
                <div style="background:#e2e8f0;border-radius:999px;height:10px;overflow:hidden;margin-top:6px;">
                  <div style="width:{accuracy}%;height:100%;background:{bar_c};border-radius:999px;"></div>
                </div>
                <div style="font-size:0.75rem;color:#64748b;text-align:right;">{accuracy:.1f}%</div>
                """, unsafe_allow_html=True)
else:
    st.info("No topic data yet. Complete some quizzes or flashcard sessions to see your progress.")

st.divider()

# ─── Recent Sessions ───────────────────────────────────────────────────────────
st.subheader("🕐 Recent Study Sessions")

sessions = prog.get("recent_sessions", [])
if sessions:
    for s in sessions:
        started = s.get("started_at", "")[:16].replace("T", " ") if s.get("started_at") else "Unknown"
        acc_s = s.get("accuracy", 0)
        acc_color = "#22c55e" if acc_s >= 70 else "#f59e0b" if acc_s >= 50 else "#ef4444"
        with st.container():
            sc1, sc2, sc3, sc4 = st.columns([2, 1, 1, 1])
            sc1.markdown(f"🕐 `{started}`")
            sc2.markdown(f"**{s.get('total_questions', 0)}** Qs")
            sc3.markdown(f"**{s.get('correct', 0)}** ✅")
            sc4.markdown(f'<span style="color:{acc_color};font-weight:700;">{acc_s:.1f}%</span>', unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.3rem 0;border-color:#f1f5f9;'>", unsafe_allow_html=True)
else:
    st.info("No study sessions yet. Take a quiz or review flashcards to track your progress.")

# ─── Recommendations ───────────────────────────────────────────────────────────
weak_topics = prog.get("weak_topics", [])
total_attempts = prog.get("total_attempts", 0)

if weak_topics or total_attempts == 0:
    st.divider()
    st.subheader("💡 Study Recommendations")
    
    if total_attempts == 0:
        st.info("Start by taking a **📝 MCQ Quiz** or reviewing **🃏 Flashcards** to populate your progress data.")
    else:
        recs = []
        if weak_topics:
            recs.append(f"🔴 Revise these weak topics: **{', '.join(weak_topics[:3])}**")
        if prog.get("overall_accuracy", 0) < 60:
            recs.append("📖 Re-read the relevant chapters before attempting more quizzes.")
        if total_attempts < 20:
            recs.append("📝 Attempt more MCQs to build a solid performance baseline.")
        
        for rec in recs:
            st.markdown(f"- {rec}")

# ─── User Stats Summary ───────────────────────────────────────────────────────
with st.expander("📊 Your Overall Stats (All Documents)"):
    try:
        headers = get_api_headers()
        resp = requests.get(
            f"{API_URL.replace('/api/v1', '')}/api/v1/user/stats",
            headers=headers,
            timeout=5
        )
        if resp.status_code == 200:
            stats = resp.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("📄 Documents", stats.get("documents", {}).get("count", 0))
            col2.metric("💾 Cache Size", f"{stats.get('cache', {}).get('total_items', 0)} items")
            col3.metric("📁 Storage", stats.get("storage_path", "N/A")[:30] + "...")
        else:
            st.info("Login to see your overall stats")
    except Exception:
        st.info("Login to see your overall stats")