"""
MediTutor AI — Page 4: MCQ Quiz with User Isolation
"""

import streamlit as st
import requests
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="MCQ Quiz — MediTutor AI", page_icon="📝", layout="wide")

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
.option-default { background:#f8fafc; border:2px solid #e2e8f0; border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; cursor:pointer; }
.option-selected { background:#eff6ff; border:2px solid #3b82f6; border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; }
.option-correct  { background:#f0fdf4; border:2px solid #22c55e; border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; }
.option-wrong    { background:#fff1f2; border:2px solid #ef4444; border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; }
.explanation-box { background:#fffbeb; border-left:4px solid #f59e0b; padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin-top:0.5rem; font-size:0.9rem; }
.score-big { font-size:3rem; font-weight:800; text-align:center; }
.user-badge { background: #e2e8f0; border-radius: 6px; padding: 0.2rem 0.5rem; font-size: 0.7rem; font-family: monospace; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("📝 MCQ Quiz")
st.caption("Test your knowledge with AI-generated multiple-choice questions.")

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

# ─── Ensure session with headers ────────────────────────────────────────────────
if not st.session_state.get("session_id"):
    try:
        headers = get_api_headers()
        # No student_id in body — backend uses X-User-ID header
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

# ─── Generation Controls ───────────────────────────────────────────────────────
if not st.session_state.get("current_mcqs") or st.session_state.get("quiz_submitted"):
    with st.expander("⚙️ Quiz Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Topic (optional)", placeholder="e.g. Pharmacokinetics")
        with col2:
            count = st.slider("Number of questions", 3, 20, 5)

        if st.button("🎯 Generate Quiz", type="primary", use_container_width=True):
            with st.spinner("🤖 Creating quiz questions..."):
                try:
                    headers = get_api_headers()
                    payload = {
                        "document_id": doc_id,
                        "count": count,
                        "topic": topic.strip() if topic.strip() else None,
                    }
                    resp = requests.post(
                        f"{API_URL}/mcq/generate",
                        json=payload,
                        headers=headers,
                        timeout=120
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["current_mcqs"] = data["questions"]
                        st.session_state["mcq_answers"] = {}
                        st.session_state["quiz_submitted"] = False
                        st.session_state["quiz_results"] = None
                        model = data.get("model_used", "")
                        st.success(f"✅ {data['total_generated']} questions ready! — `{model}`")
                        st.rerun()
                    elif resp.status_code == 401:
                        st.error("❌ Authentication error. Please refresh the page.")
                    else:
                        st.error(f"❌ {resp.json().get('detail', 'Failed')}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Timed out. Try again.")
                except Exception as e:
                    st.error(f"❌ {e}")

# ─── Quiz Interface ────────────────────────────────────────────────────────────
questions = st.session_state.get("current_mcqs", [])
submitted = st.session_state.get("quiz_submitted", False)
results_data = st.session_state.get("quiz_results")

if questions:
    st.divider()

    if submitted and results_data:
        # ── Results View ──────────────────────────────────────────────────────
        score = results_data["score"]
        correct = results_data["correct"]
        total = results_data["total"]

        # Score display
        color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"
        emoji = "🏆" if score >= 70 else "📚" if score >= 50 else "💪"
        st.markdown(
            f'<div class="score-big" style="color:{color}">{emoji} {score:.0f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"<p style='text-align:center; color:#64748b;'>{correct} / {total} correct</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Questions", total)
        col2.metric("Correct", correct)
        col3.metric("Score", f"{score:.1f}%")

        # Weak topics
        feedback = results_data.get("feedback", [])
        wrong = [f for f in feedback if not f["is_correct"]]
        if wrong:
            weak = list({f["topic"] for f in wrong})
            st.warning(f"📌 **Weak topics identified:** {', '.join(weak)}")

        st.divider()
        st.subheader("📋 Question Review")

        for i, fb in enumerate(feedback, 1):
            is_correct = fb["is_correct"]
            icon = "✅" if is_correct else "❌"
            with st.expander(f"{icon} Q{i}: {fb['question'][:80]}..."):
                opts = [q for q in questions if q["id"] == fb.get("question_id", "")]
                if opts:
                    q_opts = opts[0]["options"]
                    for j, opt in enumerate(q_opts):
                        if j == fb["correct_index"] and j == fb["selected_index"]:
                            css = "option-correct"
                            tag = " ✅ (Your answer — Correct!)"
                        elif j == fb["correct_index"]:
                            css = "option-correct"
                            tag = " ✅ (Correct answer)"
                        elif j == fb["selected_index"]:
                            css = "option-wrong"
                            tag = " ❌ (Your answer)"
                        else:
                            css = "option-default"
                            tag = ""
                        st.markdown(
                            f'<div class="{css}">{chr(65+j)}. {opt}{tag}</div>',
                            unsafe_allow_html=True,
                        )

                if fb.get("explanation"):
                    st.markdown(
                        f'<div class="explanation-box">💡 <b>Explanation:</b> {fb["explanation"]}</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        if st.button("🔄 Start New Quiz", type="primary", use_container_width=True):
            st.session_state["current_mcqs"] = []
            st.session_state["quiz_submitted"] = False
            st.session_state["quiz_results"] = None
            st.session_state["mcq_answers"] = {}
            st.rerun()

    else:
        # ── Quiz Taking View ──────────────────────────────────────────────────
        answered = len(st.session_state.get("mcq_answers", {}))
        st.markdown(f"**Progress:** {answered}/{len(questions)} answered")
        st.progress(answered / len(questions) if questions else 0)
        st.markdown("<br>", unsafe_allow_html=True)

        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}. {q['question']}**")
            
            options = q["options"]
            current_answer = st.session_state["mcq_answers"].get(q["id"])
            labels = [f"{chr(64+j+1)}. {opt}" for j, opt in enumerate(options)]
            
            selected = st.radio(
                f"Select answer for Q{i}",
                options=range(len(options)),
                format_func=lambda x: labels[x],
                key=f"mcq_{q['id']}",
                index=current_answer if current_answer is not None else None,
                label_visibility="collapsed",
            )
            
            if selected is not None:
                st.session_state["mcq_answers"][q["id"]] = selected

            st.markdown("---")

        # Submit button
        all_answered = len(st.session_state["mcq_answers"]) == len(questions)
        if not all_answered:
            st.warning(f"⚠️ Please answer all {len(questions)} questions before submitting.")

        if st.button(
            "📤 Submit Quiz",
            type="primary",
            disabled=not all_answered,
            use_container_width=True,
        ):
            with st.spinner("Grading your answers..."):
                try:
                    headers = get_api_headers()
                    answers_payload = [
                        {
                            "question_id": qid,
                            "selected_index": sel,
                            "topic": next((q["topic"] for q in questions if q["id"] == qid), "General"),
                        }
                        for qid, sel in st.session_state["mcq_answers"].items()
                    ]
                    
                    submit_payload = {
                        "document_id": doc_id,
                        "session_id": st.session_state.get("session_id", ""),
                        "answers": answers_payload,
                    }
                    
                    resp = requests.post(
                        f"{API_URL}/mcq/submit",
                        json=submit_payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        # Attach question_id to feedback
                        for fb, ans in zip(result["feedback"], answers_payload):
                            fb["question_id"] = ans["question_id"]
                        st.session_state["quiz_results"] = result
                        st.session_state["quiz_submitted"] = True
                        st.rerun()
                    elif resp.status_code == 401:
                        st.error("❌ Authentication error. Please refresh the page.")
                    else:
                        st.error(f"Submission failed: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error: {e}")