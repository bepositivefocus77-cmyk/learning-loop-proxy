"""
AI Assistant Manager (SMB-friendly dashboard, multi-user version)
----------------------------------------------------------------------
A simplified, plain-language dashboard for a small team's designated
reviewer/admin. Now requires login and only shows the logged-in
account's own data, via the authenticated multi-user proxy.

Run with:
    streamlit run smb_dashboard.py
(the auth-enabled proxy must also be running:
    uvicorn learning_loop_proxy_auth:app --reload)
"""

import pandas as pd
import requests
import streamlit as st

PROXY_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Assistant Manager", layout="wide")

# ---------------------------------------------------------------------------
# Login / signup gate
# ---------------------------------------------------------------------------
if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.email = None

if st.session_state.token is None:
    st.title("🤖 AI Assistant Manager")
    st.caption("Log in to see what your team has been asking your AI assistant.")

    tab_login, tab_signup = st.tabs(["Log in", "Create an account"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                try:
                    r = requests.post(f"{PROXY_URL}/login", json={"email": email, "password": password}, timeout=10)
                    if r.status_code == 200:
                        st.session_state.token = r.json()["access_token"]
                        st.session_state.email = email
                        st.rerun()
                    else:
                        st.error(r.json().get("detail", "Login failed."))
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the AI assistant service. Make sure it's running.")

    with tab_signup:
        with st.form("signup_form"):
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password (min 8 characters)", type="password", key="signup_password")
            submitted = st.form_submit_button("Create account")
            if submitted:
                try:
                    r = requests.post(
                        f"{PROXY_URL}/signup", json={"email": new_email, "password": new_password}, timeout=10
                    )
                    if r.status_code == 200:
                        st.session_state.token = r.json()["access_token"]
                        st.session_state.email = new_email
                        st.rerun()
                    else:
                        st.error(r.json().get("detail", "Could not create account."))
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the AI assistant service. Make sure it's running.")

    st.stop()


# ---------------------------------------------------------------------------
# Logged in -- everything below only runs for an authenticated user
# ---------------------------------------------------------------------------
def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


st.title("🤖 AI Assistant Manager")
col_title, col_logout = st.columns([5, 1])
with col_title:
    st.caption(f"Logged in as {st.session_state.email} — this view shows only your own team's data.")
with col_logout:
    if st.button("Log out"):
        st.session_state.token = None
        st.session_state.email = None
        st.rerun()


def load_data():
    try:
        r = requests.get(f"{PROXY_URL}/export", headers=auth_headers(), timeout=15)
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the AI assistant service. Make sure it's running.")
        st.stop()

    if r.status_code == 401:
        st.session_state.token = None
        st.warning("Your session expired. Please log in again.")
        st.rerun()

    if r.status_code != 200:
        st.error(f"Could not load data ({r.status_code}).")
        st.stop()

    entries = r.json().get("entries", [])
    df = pd.DataFrame(entries)

    if df.empty:
        return df

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


df = load_data()

if df.empty:
    st.info("No questions have been asked yet. Once your team starts using the AI assistant, activity will show up here.")
    st.stop()

# ---------------------------------------------------------------------------
# Top-level, plain-language metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Questions Asked", len(df))
col2.metric("Answers Approved", int((df["feedback"] == "correct").sum()))
col3.metric("Answers Corrected", int((df["feedback"] == "incorrect").sum()))
col4.metric("Needs Your Review", int((df["feedback"].isna() | (df["feedback"] == "pending")).sum()))

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

pending_df = df[df["feedback"].isna() | (df["feedback"] == "pending")].sort_values("timestamp", ascending=False)

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
                    r = requests.post(
                        f"{PROXY_URL}/correct",
                        json={"interaction_id": row["interaction_id"], "is_correct": True},
                        headers=auth_headers(),
                        timeout=10,
                    )
                    if r.status_code == 200:
                        st.success("Approved. Refresh the page to update the list.")
                    else:
                        st.error(f"Something went wrong ({r.status_code}).")

            with col_b:
                correction_text = st.text_area(
                    "If this answer was wrong, write the correct answer here:",
                    key=f"correction_{row['interaction_id']}",
                )
                if st.button("🚩 Flag as incorrect", key=f"flag_{row['interaction_id']}"):
                    r = requests.post(
                        f"{PROXY_URL}/correct",
                        json={
                            "interaction_id": row["interaction_id"],
                            "is_correct": False,
                            "correction": correction_text or None,
                        },
                        headers=auth_headers(),
                        timeout=10,
                    )
                    if r.status_code == 200:
                        st.success("Flagged and saved. Refresh the page to update the list.")
                    else:
                        st.error(f"Something went wrong ({r.status_code}).")

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
    status_counts = df["feedback"].fillna("pending").map(status_labels).fillna(df["feedback"]).value_counts()
    st.bar_chart(status_counts)

st.divider()

# ---------------------------------------------------------------------------
# Full history + download
# ---------------------------------------------------------------------------
st.subheader("📜 Full History")

display_df = df.copy()
display_df["Question"] = display_df["prompt"].str.slice(0, 60) + "..."
display_df["Answered By"] = display_df["provider"].apply(friendly_provider)
display_df["Status"] = display_df["feedback"].fillna("pending").map(
    {"correct": "Approved", "incorrect": "Corrected", "pending": "Needs Review"}
).fillna(display_df["feedback"])
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