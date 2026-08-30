"""
AI Assistant Manager (SMB-friendly dashboard)
------------------------------------------------
A simplified, plain-language dashboard for a small team's designated
reviewer/admin. Reuses the full Learning Loop Proxy backend (multi-provider
chat, Pinecone-backed retrieval, consent-boundary tagging) but hides the
technical machinery behind everyday language, so a non-technical admin can:

  - See what the team has been asking the AI assistant
  - Approve good answers or flag and correct bad ones, in one click
  - See a plain-language warning whenever sensitive info was sent outside
  - Download the whole history whenever they want

This dashboard talks to the running proxy's API (default localhost:8000)
rather than reading the log file directly, so approvals/flags take effect
immediately through the same /correct endpoint used elsewhere.

Run with:
    streamlit run smb_dashboard.py
(the proxy must also be running: uvicorn learning_loop_proxy_min:app --reload)
"""

import json
import os

import pandas as pd
import requests
import streamlit as st

LOG_FILE = "learning_log.jsonl"
PROXY_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Assistant Manager", layout="wide")
st.title("🤖 AI Assistant Manager")
st.caption("A simple view of what your team has been asking your AI assistant, and a record you own and control.")


def load_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    rows = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    for col in ["provider", "feedback", "correction", "sensitivity", "policy_warning"]:
        if col not in df.columns:
            df[col] = None
    df["provider"] = df["provider"].fillna("unknown")
    df["feedback"] = df["feedback"].fillna("pending")
    df["sensitivity"] = df["sensitivity"].fillna("unclassified")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


PROVIDER_LABELS = {"groq": "Groq AI", "gemini": "Google Gemini", "unknown": "AI Service"}
SENSITIVITY_LABELS = {
    "public": "🌐 Public — safe to share anywhere",
    "internal": "🏢 Internal — team use only",
    "confidential": "🔒 Confidential — sensitive info",
    "unclassified": "⚪ Not labeled",
}


def friendly_provider(p):
    return PROVIDER_LABELS.get(p, p)


def friendly_sensitivity(s):
    return SENSITIVITY_LABELS.get(s, s)


df = load_log()

if df.empty:
    st.warning("No questions have been asked yet. Once your team starts using the AI assistant, activity will show up here.")
    st.stop()

# ---------------------------------------------------------------------------
# Top-level, plain-language metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Questions Asked", len(df))
col2.metric("Answers Approved", int((df["feedback"] == "correct").sum()))
col3.metric("Answers Corrected", int((df["feedback"] == "incorrect").sum()))
col4.metric("Needs Your Review", int((df["feedback"] == "pending").sum()))

sensitive_sent = df[(df["sensitivity"] == "confidential") & (df["policy_warning"].notna())]
if len(sensitive_sent) > 0:
    st.warning(
        f"⚠️ {len(sensitive_sent)} question(s) marked confidential were sent to an outside AI service. "
        "Review these before reusing or sharing the answers."
    )

st.divider()

# ---------------------------------------------------------------------------
# Needs Review — actionable queue for the admin
# ---------------------------------------------------------------------------
st.subheader("📋 Needs Your Review")

pending_df = df[(df["feedback"] == "pending") & df["interaction_id"].notna()].sort_values("timestamp", ascending=False)

if pending_df.empty:
    st.success("Nothing waiting on you right now — you're all caught up.")
else:
    for _, row in pending_df.iterrows():
        with st.expander(f"❓ {row['prompt'][:80]}"):
            st.write("**Question asked:**", row["prompt"])
            st.write("**AI's answer:**", row["response"])
            st.write("**Answered by:**", friendly_provider(row["provider"]))
            st.write("**Sensitivity:**", friendly_sensitivity(row["sensitivity"]))
            if row.get("policy_warning"):
                st.info(f"ℹ️ {row['policy_warning']}")

            col_a, col_b = st.columns([1, 2])
            with col_a:
                if st.button("✅ Approve this answer", key=f"approve_{row['interaction_id']}"):
                    try:
                        r = requests.post(
                            f"{PROXY_URL}/correct",
                            json={"interaction_id": row["interaction_id"], "is_correct": True},
                            timeout=10,
                        )
                        if r.status_code == 200:
                            st.success("Approved. Refresh the page to update the list.")
                        else:
                            st.error(f"Something went wrong ({r.status_code}). Is the assistant running?")
                    except requests.exceptions.ConnectionError:
                        st.error("Can't reach the AI assistant. Make sure it's running.")

            with col_b:
                correction_text = st.text_area("If this answer was wrong, write the correct answer here:", key=f"correction_{row['interaction_id']}")
                if st.button("🚩 Flag as incorrect", key=f"flag_{row['interaction_id']}"):
                    try:
                        r = requests.post(
                            f"{PROXY_URL}/correct",
                            json={
                                "interaction_id": row["interaction_id"],
                                "is_correct": False,
                                "correction": correction_text or None,
                            },
                            timeout=10,
                        )
                        if r.status_code == 200:
                            st.success("Flagged and saved. Refresh the page to update the list.")
                        else:
                            st.error(f"Something went wrong ({r.status_code}). Is the assistant running?")
                    except requests.exceptions.ConnectionError:
                        st.error("Can't reach the AI assistant. Make sure it's running.")

st.divider()

# ---------------------------------------------------------------------------
# Simple charts
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Which AI service answered")
    provider_counts = df["provider"].apply(friendly_provider).value_counts()
    st.bar_chart(provider_counts)

with chart_col2:
    st.subheader("Answer review status")
    status_labels = {"correct": "Approved", "incorrect": "Corrected", "pending": "Needs Review"}
    status_counts = df["feedback"].map(status_labels).fillna(df["feedback"]).value_counts()
    st.bar_chart(status_counts)

st.divider()

# ---------------------------------------------------------------------------
# Full history + download
# ---------------------------------------------------------------------------
st.subheader("📜 Full History")

display_df = df.copy()
display_df["Question"] = display_df["prompt"].str.slice(0, 60) + "..."
display_df["Answered By"] = display_df["provider"].apply(friendly_provider)
display_df["Status"] = display_df["feedback"].map({"correct": "Approved", "incorrect": "Corrected", "pending": "Needs Review"}).fillna(display_df["feedback"])
display_df["Sensitivity"] = display_df["sensitivity"].apply(friendly_sensitivity)

show_cols = ["timestamp", "Question", "Answered By", "Status", "Sensitivity"]
st.dataframe(
    display_df[show_cols].sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Download your team's full AI history",
    data=df.to_csv(index=False),
    file_name="team_ai_history.csv",
    mime="text/csv",
    help="This is your data. Download it anytime — it doesn't depend on any AI provider.",
)