import streamlit as st
import pandas as pd
import anthropic
import json

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Backlog Prioritizer",
    page_icon="🎯",
    layout="wide"
)

# ─── Initialize Anthropic Client ────────────────────────────────
required_keys = ["ANTHROPIC_API_KEY"]
missing = [k for k in required_keys if k not in st.secrets]
if missing:
    st.error(
        f"Missing secret(s): {', '.join(missing)}. "
        "Add them in Streamlit Cloud → Settings → Secrets "
        "(or .streamlit/secrets.toml when running locally)."
    )
    st.stop()

client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ─── Load Backlog CSV ────────────────────────────────────────────
@st.cache_data
def load_backlog(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Filter to Stories only — skip Epics
    df = df[df["Issue Type"] == "Story"].reset_index(drop=True)
    return df

# ─── RICE Scoring with Claude ────────────────────────────────────
def prioritize_backlog(df: pd.DataFrame, context: str = "") -> str:
    """
    PO Decision: Use RICE scoring (Reach, Impact, Confidence, Effort)
    instead of MoSCoW because RICE is quantifiable and defensible.
    Claude scores each story and explains its reasoning.
    Human-in-the-loop: PO can challenge any score.
    """
    # Build story list for Claude
    stories = []
    for _, row in df.iterrows():
        stories.append({
            "id": row.get("Issue Type", "Story"),
            "summary": row.get("Summary", ""),
            "epic": row.get("Epic Name", ""),
            "priority": row.get("Priority", ""),
            "points": row.get("Story Points", ""),
            "sprint": row.get("Sprint", ""),
            "description": str(row.get("Description", ""))[:300]
        })

    stories_json = json.dumps(stories, indent=2)

    prompt = f"""You are an expert Product Owner helping prioritize a 
software development backlog using the RICE scoring framework.

RICE Scoring:
- Reach (1-10): How many users/processes does this impact?
- Impact (1-10): How significantly does it improve outcomes?
- Confidence (1-10): How certain are we about estimates?
- Effort (1-10): How much work is required? (higher = more effort)
- RICE Score = (Reach × Impact × Confidence) / Effort

Additional PO context: {context if context else "None provided"}

Here are the backlog stories to prioritize:
{stories_json}

Please:
1. Score each story using RICE (1-10 for each dimension)
2. Calculate the RICE score
3. Rank stories from highest to lowest RICE score
4. Provide a 1-sentence reasoning for each story's ranking

Respond ONLY in this exact JSON format, no preamble:
{{
  "prioritized_stories": [
    {{
      "summary": "story summary here",
      "epic": "epic name here",
      "reach": 8,
      "impact": 9,
      "confidence": 7,
      "effort": 5,
      "rice_score": 10.08,
      "rank": 1,
      "reasoning": "One sentence explanation here"
    }}
  ]
}}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

# ─── Parse Claude Response ───────────────────────────────────────
def parse_response(response_text: str) -> pd.DataFrame:
    try:
        # Clean response
        clean = response_text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]

        data = json.loads(clean)
        stories = data.get("prioritized_stories", [])
        df = pd.DataFrame(stories)

        # Round RICE score
        df["rice_score"] = df["rice_score"].round(2)

        # Reorder columns
        cols = ["rank", "summary", "epic", "reach", "impact",
                "confidence", "effort", "rice_score", "reasoning"]
        df = df[[c for c in cols if c in df.columns]]

        return df.sort_values("rank").reset_index(drop=True)

    except Exception as e:
        st.error(f"Error parsing response: {str(e)}")
        return pd.DataFrame()

# ─── UI Layout ───────────────────────────────────────────────────
st.title("🎯 AI Backlog Prioritizer")
st.markdown(
    "*Upload your Jira backlog CSV and let Claude prioritize "
    "it using RICE scoring — then challenge any decision.*"
)
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 How RICE Works")
    st.markdown("""
**Reach** — How many users impacted?
**Impact** — How much does it improve things?
**Confidence** — How sure are we?
**Effort** — How hard is it to build?

**Score = (R × I × C) / E**

Higher score = higher priority
    """)
    st.divider()
    st.markdown("### 💡 Tips")
    st.markdown("""
- Add PO context to guide scoring
- Challenge any ranking in the chat
- Export final table to CSV
    """)

# ─── File Upload OR Default CSV ──────────────────────────────────
st.markdown("### 📂 Backlog Source")
use_default = st.checkbox(
    "Use sample CLMS Healthcare Claims backlog",
    value=True
)

if use_default:
    df = load_backlog("data/CLMS_Jira_Import.csv")
    st.success(
        f"✅ Loaded CLMS backlog — "
        f"{len(df)} stories across "
        f"{df['Epic Name'].nunique()} epics"
    )
else:
    uploaded = st.file_uploader(
        "Upload your Jira CSV export",
        type="csv"
    )
    if uploaded:
        df = pd.read_csv(uploaded)
        df = df[df["Issue Type"] == "Story"].reset_index(drop=True)
        st.success(f"✅ Loaded {len(df)} stories")
    else:
        st.info("Upload a CSV or use the sample backlog above")
        st.stop()

# ─── Show Raw Backlog ────────────────────────────────────────────
with st.expander("👀 View Raw Backlog", expanded=False):
    st.dataframe(
        df[["Summary", "Epic Name", "Priority",
            "Story Points", "Sprint"]],
        use_container_width=True
    )

# ─── PO Context Input ────────────────────────────────────────────
st.markdown("### 🧠 Add PO Context (Optional)")
context = st.text_area(
    "Tell Claude what matters most for this sprint:",
    placeholder=(
        "e.g. We are 6 weeks from go-live on Jul 30 2026. "
        "Data quality and ingestion stories are critical path. "
        "Dashboard features are nice-to-have for v1."
    ),
    height=100
)

# ─── Prioritize Button ───────────────────────────────────────────
st.markdown("### 🚀 Prioritize Backlog")

if "prioritized_df" not in st.session_state:
    st.session_state.prioritized_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "raw_response" not in st.session_state:
    st.session_state.raw_response = None

if st.button("⚡ Prioritize with AI", type="primary"):
    with st.spinner(
        "Claude is scoring your backlog using RICE framework..."
    ):
        raw = prioritize_backlog(df, context)
        st.session_state.raw_response = raw
        result_df = parse_response(raw)
        if not result_df.empty:
            st.session_state.prioritized_df = result_df
            st.session_state.chat_history = []

# ─── Show Prioritized Results ────────────────────────────────────
if st.session_state.prioritized_df is not None:
    st.divider()
    st.markdown("### 📊 Prioritized Backlog")

    result_df = st.session_state.prioritized_df

    # Color code by rank
    st.dataframe(
        result_df,
        use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "summary": st.column_config.TextColumn("Story", width="large"),
            "epic": st.column_config.TextColumn("Epic", width="medium"),
            "reach": st.column_config.NumberColumn("Reach", width="small"),
            "impact": st.column_config.NumberColumn("Impact", width="small"),
            "confidence": st.column_config.NumberColumn("Conf.", width="small"),
            "effort": st.column_config.NumberColumn("Effort", width="small"),
            "rice_score": st.column_config.NumberColumn(
                "RICE Score", width="small"
            ),
            "reasoning": st.column_config.TextColumn(
                "Reasoning", width="large"
            ),
        }
    )

    # ─── Export Button ───────────────────────────────────────────
    csv_export = result_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Prioritized Backlog to CSV",
        data=csv_export,
        file_name="prioritized_backlog.csv",
        mime="text/csv"
    )

    # ─── Challenge / Chat Section ────────────────────────────────
    st.divider()
    st.markdown("### 💬 Challenge the Prioritization")
    st.markdown(
        "*Ask Claude why it ranked something a certain way, "
        "or ask it to re-rank based on new context.*"
    )

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if challenge := st.chat_input(
        "e.g. Why is the KPI Dashboard ranked so low? "
        "Move FTP Error Handling higher — it's a blocker..."
    ):
        with st.chat_message("user"):
            st.markdown(challenge)

        st.session_state.chat_history.append({
            "role": "user",
            "content": challenge
        })

        with st.chat_message("assistant"):
            with st.spinner("Re-evaluating..."):
                # Build context for challenge
                current_ranking = result_df[
                    ["rank", "summary", "rice_score", "reasoning"]
                ].to_string()

                challenge_prompt = f"""You are an expert Product Owner assistant.

The current RICE-prioritized backlog is:
{current_ranking}

The PO is challenging the prioritization with this input:
"{challenge}"

Please:
1. Acknowledge their concern
2. Explain your original reasoning
3. If they want a re-rank, provide updated rankings with explanation
4. Keep your answer concise and practical

Respond in plain English — not JSON."""

                challenge_response = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": challenge_prompt}
                    ]
                )

                answer = challenge_response.content[0].text
                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

# ─── Footer ──────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by Padmini Nagarajan │ AI Product Owner Portfolio │ "
    "RICE Prioritization powered by Anthropic Claude"
)
