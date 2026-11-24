"""Yojana Saathi — Streamlit UI.

Set API_URL to use a deployed API service; leave it empty to run everything in-process.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import httpx
import streamlit as st

from saathi.client import ClientError, HttpClient, LocalClient, SaathiClient
from saathi.eligibility import FLAG_LABELS
from saathi.llm import LANGUAGES

st.set_page_config(
    page_title="Yojana Saathi",
    page_icon="🤝",
    layout="wide",
    menu_items={"About": "Yojana Saathi — find welfare schemes you may be eligible for."},
)

log = logging.getLogger("saathi.ui")
API_URL = os.getenv("API_URL", "").strip()
UI_API_KEY = os.getenv("UI_API_KEY", "")
STATES = [
    "Karnataka",
    "Andhra Pradesh",
    "Bihar",
    "Delhi",
    "Gujarat",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Rajasthan",
    "Tamil Nadu",
    "Telangana",
    "Uttar Pradesh",
    "West Bengal",
    "Other",
]
UNSAID = "Prefer not to say"
VERDICT_TITLES = {
    "eligible": "✅ Likely eligible",
    "maybe": "❔ Possibly eligible — add more details",
    "not_eligible": "❌ Probably not eligible",
}


@st.cache_resource(show_spinner="Loading scheme knowledge base…")
def _local_client() -> LocalClient:
    return LocalClient()


def get_client() -> SaathiClient:
    if not API_URL:
        return _local_client()
    if "client" not in st.session_state:
        st.session_state.client = HttpClient(API_URL, UI_API_KEY, session_id=uuid.uuid4().hex)
    return st.session_state.client


def friendly(e: Exception) -> str:
    if isinstance(e, ClientError) and e.status == 429:
        return "You're asking very quickly — please wait a minute and try again."
    if isinstance(e, ClientError) and e.status == 503:
        return "The service is starting up. Please try again in a few seconds."
    if isinstance(e, httpx.HTTPError):
        return "Can't reach the Saathi service right now. Please try again shortly."
    return "Something went wrong. Please try again."


client = get_client()

# ---------------- Sidebar: citizen profile ----------------
with st.sidebar:
    st.header("👤 Your details")
    st.caption("Used only to check eligibility. Not stored.")
    age = st.number_input("Age", min_value=0, max_value=110, value=None, placeholder="e.g. 34")
    gender = st.selectbox("Gender", [UNSAID, "female", "male", "transgender"])
    state = st.selectbox("State", [UNSAID, *STATES])
    residence = st.radio("Area", [UNSAID, "rural", "urban"], horizontal=True)
    category = st.selectbox("Social category", [UNSAID, "general", "obc", "sc", "st"])
    income = st.number_input("Annual family income (₹)", min_value=0, step=10000, value=None, placeholder="e.g. 180000")
    st.markdown("**Tick all that apply**")
    flags = [f for f, label in FLAG_LABELS.items() if st.checkbox(label, key=f"flag_{f}")]
    st.divider()
    lang_code = st.selectbox("Answer language", list(LANGUAGES), format_func=LANGUAGES.get)


def _opt(v: Any) -> Any:
    return None if v == UNSAID else v


profile: dict[str, Any] = {
    "age": int(age) if age is not None else None,
    "gender": _opt(gender),
    "state": _opt(state),
    "residence": _opt(residence),
    "social_category": _opt(category),
    "annual_income": int(income) if income is not None else None,
    "flags": flags,
}

# ---------------- Header ----------------
st.title("🤝 Yojana Saathi")
st.markdown("Find government welfare schemes you may be eligible for, and ask questions in **English, हिन्दी or ಕನ್ನಡ**.")
try:
    h = client.health()
    st.caption(
        f"{h['schemes']} schemes · retrieval: {h['retrieval']} · "
        f"{'AI answers' if h['llm_configured'] else 'offline answers (no LLM key)'} · v{h['version']}"
    )
except Exception as e:  # noqa: BLE001 - surface any backend failure as a friendly banner
    st.warning(friendly(e))

tab_find, tab_chat = st.tabs(["🔍 Find my schemes", "💬 Ask Saathi"])

# ---------------- Tab 1: eligibility ----------------
with tab_find:
    try:
        data = client.eligibility(profile)
    except Exception as e:  # noqa: BLE001
        st.error(friendly(e))
        data = {"results": [], "counts": {}}

    counts = data["counts"]
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Likely eligible", counts.get("eligible", 0))
    c2.metric("❔ Need more info", counts.get("maybe", 0))
    c3.metric("❌ Not eligible", counts.get("not_eligible", 0))

    for verdict, title in VERDICT_TITLES.items():
        group = [r for r in data["results"] if r["verdict"] == verdict]
        if not group:
            continue
        st.subheader(title)
        for r in group:
            s = r["scheme"]
            with st.expander(f"{s['name']}  ·  {s['level']}  ·  {s['category']}", expanded=verdict == "eligible"):
                st.write(s["summary"])
                st.markdown(f"**Benefits:** {s['benefits']}")
                if r["reasons"]:
                    st.markdown(
                        ("**Reason:** " if verdict == "not_eligible" else "**Why:** ") + "; ".join(r["reasons"])
                    )
                if r["missing"]:
                    st.info("To confirm, add: " + ", ".join(r["missing"]))
                st.markdown("**Documents:** " + ", ".join(s["documents"]))
                st.markdown(f"**How to apply:** {s['how_to_apply']}")
                st.link_button("Official website ↗", s["official_url"])
    st.caption("⚠️ Guidance only, not an official decision. Rules change — always confirm on the official portal.")

# ---------------- Tab 2: chat ----------------
with tab_chat:
    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])

    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] != "assistant" or not msg.get("request_id"):
                continue
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"[{s['tag']}] **{s['scheme_name']}** — {s['section']} · [{s['url']}]({s['url']})")
            rating = st.feedback("thumbs", key=f"fb_{msg['request_id']}")
            if rating is not None and not msg.get("rated"):
                try:
                    client.feedback(
                        {
                            "request_id": msg["request_id"],
                            "rating": 1 if rating == 1 else -1,
                            "question": history[i - 1]["content"],
                            "answer": msg["content"],
                        }
                    )
                    st.toast("Thanks for the feedback!")
                except Exception as e:  # noqa: BLE001 - feedback is best-effort, never block the user
                    log.warning("feedback_failed: %s", e)
                msg["rated"] = True

    examples = [
        "What documents do I need for PM-KISAN?",
        "ನಾನು ಗೃಹ ಲಕ್ಷ್ಮಿ ಯೋಜನೆಗೆ ಹೇಗೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಬೇಕು?",
        "मुझे मुफ्त इलाज के लिए कौन सी योजना मिल सकती है?",
    ]
    cols = st.columns(len(examples))
    clicked = next((q for c, q in zip(cols, examples, strict=True) if c.button(q, use_container_width=True)), None)
    question = st.chat_input("Ask about any scheme…", max_chars=1000) or clicked

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        payload = {
            "question": question,
            "language": lang_code,
            "profile": profile if any(v not in (None, []) for v in profile.values()) else None,
            "history": [{"role": m["role"], "content": m["content"]} for m in history][-6:],
        }
        meta: dict[str, Any] = {}

        def tokens() -> Any:
            for event, data in client.ask_stream(payload):
                if event == "meta":
                    meta.update(data)
                elif event == "token":
                    yield data["text"]
                elif event == "error":
                    yield f"\n\n⚠️ {data['message']}"

        with st.chat_message("assistant"):
            try:
                text = st.write_stream(tokens())
            except Exception as e:  # noqa: BLE001
                text = friendly(e)
                st.error(text)
        history.append({"role": "user", "content": question})
        history.append(
            {
                "role": "assistant",
                "content": text if isinstance(text, str) else str(text),
                "sources": meta.get("sources", []),
                "request_id": meta.get("request_id"),
            }
        )
        st.rerun()
