"""
Learning Loop Dashboard
------------------------
Reads learning_log.jsonl and shows a visual summary of your
accumulated interaction history: how many prompts went to each
provider, how many got corrections, and the full interaction table.

Run with:
    streamlit run dashboard.py
"""

import json
import os

import pandas as pd
import streamlit as st

LOG_FILE = "learning_log.jsonl"

st.set_page_config(page_title="Learning Loop Dashboard", layout="wide")
st.title("🔁 Learning Loop Dashboard")
st.caption("Your own record of every AI interaction — owned locally, not left inside a vendor's logs.")


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

    # Fill in missing fields from before the schema was extended
    for col in ["provider", "feedback", "correction"]:
        if col not in df.columns:
            df[col] = None
    df["provider"] = df["provider"].fillna("unknown")
    df["feedback"] = df["feedback"].fillna("pending")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


df = load_log()

if df.empty:
    st.warning("No interactions logged yet. Send a request to your proxy's /chat endpoint first.")
    st.stop()

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total interactions", len(df))
col2.metric("Providers used", df["provider"].nunique())
col3.metric("Corrections logged", int((df["feedback"] == "incorrect").sum()))
col4.metric("Marked correct", int((df["feedback"] == "correct").sum()))

st.divider()

# ---------------------------------------------------------------------------
# Breakdown charts
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Interactions by provider")
    provider_counts = df["provider"].value_counts()
    st.bar_chart(provider_counts)

with chart_col2:
    st.subheader("Feedback status")
    feedback_counts = df["feedback"].value_counts()
    st.bar_chart(feedback_counts)

st.divider()

# ---------------------------------------------------------------------------
# Interaction history table
# ---------------------------------------------------------------------------
st.subheader("Interaction history")

display_df = df.copy()
display_df["prompt_preview"] = display_df["prompt"].str.slice(0, 60) + "..."
display_df["response_preview"] = display_df["response"].str.slice(0, 60) + "..."

show_cols = ["timestamp", "provider", "model_used", "prompt_preview", "response_preview", "feedback"]
show_cols = [c for c in show_cols if c in display_df.columns]

st.dataframe(
    display_df[show_cols].sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Full detail viewer for a single interaction
# ---------------------------------------------------------------------------
st.subheader("Inspect a single interaction")

if "interaction_id" in df.columns:
    valid_ids = df["interaction_id"].dropna().tolist()
else:
    valid_ids = []

if not valid_ids:
    st.info("No interactions with an interaction_id yet (older log entries predate this field).")
else:
    selected_id = st.selectbox("Select interaction_id", valid_ids)
    row = df[df["interaction_id"] == selected_id].iloc[0]
    st.write("**Prompt:**", row.get("prompt"))
    st.write("**Response:**", row.get("response"))
    st.write("**Provider:**", row.get("provider"), " | **Model:**", row.get("model_used"))
    st.write("**Feedback:**", row.get("feedback"))
    if row.get("correction"):
        st.write("**Correction:**", row.get("correction"))