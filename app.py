"""XYZ Finance - Customer Support Portal (Streamlit entry point).

Pages
-----
* Customer Portal      - chat-style complaint submission, AI acknowledgement, tracking
* Support Console      - search / filter complaints and update their status
* Analytics Dashboard  - KPIs, charts, trend analysis, CSV report
* AI Insights          - management insights (rule-based + optional LLM summary)
"""

from __future__ import annotations

import hmac
import os
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from complaint_desk import __version__, analytics
from complaint_desk.chains import ComplaintPipeline
from complaint_desk.config import CATEGORIES, CATEGORY_LABELS, STATUSES, Settings, get_settings
from complaint_desk.storage import ComplaintStore
from complaint_desk.validators import validate_complaint

st.set_page_config(
    page_title="XYZ Finance | Complaint Desk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- setup
def _load_streamlit_secrets() -> None:
    """Copy Streamlit Community Cloud secrets into the environment (if present)."""
    try:
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_MODEL",
            "LLM_PROVIDER",
            "OLLAMA_MODEL",
            "OLLAMA_BASE_URL",
            "LLM_TEMPERATURE",
            "COMPANY_NAME",
            "SUPPORT_PIN",
            "OFFLINE_MODE",
        ):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass  # no secrets file -> use .env / environment variables


_load_streamlit_secrets()
settings: Settings = get_settings()


@st.cache_resource
def get_store(db_path: str) -> ComplaintStore:
    return ComplaintStore(db_path)


@st.cache_resource
def get_pipeline(cache_key: tuple, _settings: Settings) -> ComplaintPipeline:
    return ComplaintPipeline(_settings)


store = get_store(str(settings.db_path))
pipeline = get_pipeline(
    (settings.ai_enabled, settings.llm_provider, settings.active_model, settings.temperature, settings.company_name),
    settings,
)

PRIORITY_COLORS = {"high": "#d92d20", "medium": "#f79009", "low": "#12b76a"}
SENTIMENT_COLORS = {"angry": "#d92d20", "frustrated": "#f79009", "neutral": "#667085", "positive": "#12b76a"}
STATUS_COLORS = {"Open": "#d92d20", "In Progress": "#f79009", "Closed": "#12b76a"}
CATEGORY_COLORS = ["#0b5fff", "#7a5af8", "#d92d20", "#12b76a"]

EXAMPLES = {
    "EMI deducted twice": "My EMI payment was deducted twice from my account.",
    "Unauthorized transaction": "There is an unauthorized transaction of ₹50,000 on my account and nobody is helping me.",
    "Refund pending": "I still have not received the refund for the extra fee that was charged last week.",
    "App login failure": "The mobile app keeps crashing and I cannot log in to make a payment.",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      div[data-testid="stMetric"] {
          background: rgba(11, 95, 255, 0.06);
          border: 1px solid rgba(11, 95, 255, 0.15);
          border-radius: 12px; padding: 0.75rem 1rem;
      }
      .badge {display:inline-block; padding:2px 10px; border-radius:999px;
              font-size:0.78rem; font-weight:600; margin-right:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------ helpers
def badge(text: str, color: str) -> str:
    """HTML pill. Only ever called with fixed enum values, never user text."""
    return f"<span class='badge' style='background:{color}22;color:{color};'>{text}</span>"


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection in exported text cells."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        out[col] = out[col].map(
            lambda v: f"'{v}" if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v
        )
    return out


def report_button(df: pd.DataFrame, key: str) -> None:
    if df.empty:
        return
    st.download_button(
        "⬇️ Download complaint report (CSV)",
        data=csv_safe(df).to_csv(index=False).encode("utf-8"),
        file_name=f"complaint_report_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        key=key,
    )


def bar_chart(df: pd.DataFrame, x: str, colors: dict[str, str] | list[str] | None = None) -> alt.Chart:
    color = alt.value("#0b5fff")
    if isinstance(colors, dict):
        color = alt.Color(
            f"{x}:N",
            scale=alt.Scale(domain=[k.title() for k in colors], range=list(colors.values())),
            legend=None,
        )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count:Q", title=None, axis=alt.Axis(tickMinStep=1)),
            color=color,
            tooltip=[x, "Count"],
        )
        .properties(height=280)
    )


# ------------------------------------------------------------------------ sidebar
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"## 🏦 {settings.company_name}")
        st.caption("Customer Support Portal")

        page = st.radio(
            "Navigation",
            ["Customer Portal", "Support Console", "Analytics Dashboard", "AI Insights"],
            label_visibility="collapsed",
        )
        st.divider()

        metrics = analytics.summary_metrics(store.list_complaints())
        st.metric("Total Complaints", metrics["total"])
        c1, c2 = st.columns(2)
        c1.metric("Fraud Cases", metrics["fraud"])
        c2.metric("Loan Cases", metrics["loan"])
        c3, c4 = st.columns(2)
        c3.metric("Billing Cases", metrics["billing"])
        c4.metric("App Issues", metrics["app_issue"])
        st.divider()

        if pipeline.ai_enabled:
            where = "local" if settings.llm_provider == "ollama" else "hosted"
            st.success(f"AI mode · {settings.active_model} ({where}, temp {settings.temperature})", icon="✅")
        else:
            st.warning("Offline demo mode · add an API key (or set LLM_PROVIDER=ollama) to enable AI", icon="⚠️")
        st.caption(f"v{__version__}")
    return page


# ---------------------------------------------------------------- page: customer
def render_message(msg: dict) -> None:
    role = msg["role"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "🏦"):
        if msg.get("error"):
            st.warning(msg["content"])
            return
        if role == "user":
            st.markdown(msg["content"])
            return

        meta = msg["meta"]
        st.markdown(
            f"✅ **Complaint registered** · ID `{meta['id']}`<br>"
            f"{badge(CATEGORY_LABELS[meta['category']], '#0b5fff')}"
            f"{badge(meta['priority'].title() + ' priority', PRIORITY_COLORS[meta['priority']])}"
            f"{badge(meta['sentiment'].title(), SENTIMENT_COLORS[meta['sentiment']])}",
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"].replace("\n", "  \n"))
        if meta.get("warning"):
            st.warning(meta["warning"])
        engine = settings.active_model if meta["mode"] == "ai" else "rule-based engine"
        st.caption(f"Submitted {meta['time']} · processed by {engine}")


def handle_submission(raw_text: str, customer_name: str) -> None:
    check = validate_complaint(raw_text)

    user_msg = {"role": "user", "content": check.cleaned if check.ok else raw_text}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    if not check.ok:
        error_msg = {"role": "assistant", "content": check.error, "error": True}
        st.session_state.messages.append(error_msg)
        render_message(error_msg)
        return

    try:
        with st.spinner("Analysing your complaint..."):
            analysis = pipeline.analyze(check.cleaned)
            complaint_id = store.add(
                complaint=check.cleaned,
                category=analysis.category,
                priority=analysis.priority,
                sentiment=analysis.sentiment,
                response=analysis.response,
                customer_name=customer_name,
                mode=analysis.mode,
            )
    except Exception:
        error_msg = {
            "role": "assistant",
            "content": "Sorry, we could not register your complaint right now. Please try again in a moment.",
            "error": True,
        }
        st.session_state.messages.append(error_msg)
        render_message(error_msg)
        return

    reply = {
        "role": "assistant",
        "content": analysis.response,
        "meta": {
            "id": complaint_id,
            "category": analysis.category,
            "priority": analysis.priority,
            "sentiment": analysis.sentiment,
            "mode": analysis.mode,
            "warning": analysis.warning,
            "time": datetime.now().strftime("%d %b %Y, %H:%M"),
        },
    }
    st.session_state.messages.append(reply)
    st.rerun()  # re-render from history so the sidebar metrics update too


def page_customer() -> None:
    st.title("How can we help you today?")
    st.caption("Describe your issue and our AI assistant will register it and acknowledge it instantly.")

    customer_name = st.sidebar.text_input("Your name (optional)", key="customer_name")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.expander("🔎 Track an existing complaint"):
        track_id = st.text_input("Complaint ID", placeholder="CMP-2026-001", key="track_id")
        if track_id.strip():
            record = store.get(track_id)
            if record:
                st.markdown(
                    f"**{record['complaint_id']}** · {badge(record['status'], STATUS_COLORS[record['status']])}"
                    f"<br>Category: {CATEGORY_LABELS[record['category']]} · "
                    f"Priority: {record['priority'].title()} · Logged: {record['created_at']}",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No complaint found with that ID.")

    st.write("**Quick examples**")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        if col.button(label, key=f"ex_{label}"):
            st.session_state.pending_example = text

    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input("Describe your complaint (e.g. 'My EMI was deducted twice')")
    if prompt is None:
        prompt = st.session_state.pop("pending_example", None)
    if prompt is not None:
        handle_submission(prompt, customer_name)


# ---------------------------------------------------------------- page: support
def page_support() -> None:
    st.title("Support Console")

    if settings.support_pin and not st.session_state.get("support_ok"):
        st.info("This area is for support agents. Enter the agent PIN to continue.")
        entered = st.text_input("Agent PIN", type="password")
        if entered:
            if hmac.compare_digest(entered, settings.support_pin):
                st.session_state.support_ok = True
                st.rerun()
            st.error("Incorrect PIN.")
        return

    f1, f2, f3, f4 = st.columns([2, 1.3, 1.3, 1.3])
    search = f1.text_input("Search", placeholder="Complaint ID, name or keyword")
    categories = f2.multiselect("Category", CATEGORIES, format_func=lambda c: CATEGORY_LABELS[c])
    priorities = f3.multiselect("Priority", ["high", "medium", "low"], format_func=str.title)
    statuses = f4.multiselect("Status", STATUSES)

    df = store.list_complaints(categories, priorities, statuses, search)
    st.caption(f"{len(df)} complaint(s) found")

    if df.empty:
        st.info("No complaints match these filters.")
        return

    view = df[["complaint_id", "created_at", "category", "priority", "sentiment", "status", "complaint"]].copy()
    view["category"] = view["category"].map(CATEGORY_LABELS)
    view.columns = ["ID", "Date", "Category", "Priority", "Sentiment", "Status", "Complaint"]
    st.dataframe(view, hide_index=True, height=340)
    report_button(df, key="csv_console")

    st.subheader("Review & update a complaint")
    selected = st.selectbox("Select complaint", df["complaint_id"].tolist())
    record = store.get(selected)
    if record:
        left, right = st.columns([3, 2])
        with left:
            st.markdown(
                f"{badge(CATEGORY_LABELS[record['category']], '#0b5fff')}"
                f"{badge(record['priority'].title() + ' priority', PRIORITY_COLORS[record['priority']])}"
                f"{badge(record['sentiment'].title(), SENTIMENT_COLORS[record['sentiment']])}",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Customer:** {record['customer_name'] or 'Not provided'}")
            st.markdown("**Complaint**")
            st.write(record["complaint"])
            st.markdown("**Acknowledgement sent**")
            st.text(record["response"])
        with right:
            with st.form("status_form"):
                new_status = st.radio("Status", STATUSES, index=STATUSES.index(record["status"]))
                if st.form_submit_button("Update status", type="primary"):
                    store.update_status(selected, new_status)
                    st.success(f"{selected} marked as {new_status}.")
                    st.rerun()


# -------------------------------------------------------------- page: dashboard
def page_dashboard() -> None:
    st.title("Analytics Dashboard")
    df = store.list_complaints()
    if df.empty:
        st.info("No complaints yet. Submit one from the Customer Portal, or load sample data with "
                "`python scripts/seed_demo_data.py`.")
        return

    m = analytics.summary_metrics(df)

    row1 = st.columns(5)
    for col, (label, key) in zip(
        row1,
        [("Total Complaints", "total"), ("Fraud Cases", "fraud"), ("Loan Cases", "loan"),
         ("Billing Cases", "billing"), ("App Issue Cases", "app_issue")],
    ):
        col.metric(label, m[key])

    row2 = st.columns(5)
    row2[0].metric("Open Cases", m["open"])
    row2[1].metric("Closed Cases", m["closed"])

    st.write("")
    left, right = st.columns(2)
    with left:
        st.subheader("Complaints by category")
        cat_df = analytics.category_counts(df)
        donut = (
            alt.Chart(cat_df)
            .mark_arc(innerRadius=70)
            .encode(
                theta="Count:Q",
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(domain=cat_df["Category"].tolist(), range=CATEGORY_COLORS),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["Category", "Count"],
            )
            .properties(height=300)
        )
        st.altair_chart(donut)
    with right:
        st.subheader("Complaints by priority")
        st.altair_chart(bar_chart(analytics.priority_counts(df), "Priority", PRIORITY_COLORS))

    left, right = st.columns(2)
    with left:
        st.subheader("Daily complaint trend")
        st.line_chart(analytics.daily_trend(df), height=280)
    with right:
        st.subheader("Customer sentiment")
        st.altair_chart(bar_chart(analytics.sentiment_counts(df), "Sentiment", SENTIMENT_COLORS))

    st.subheader("Latest complaints")
    latest = df.head(10)[["complaint_id", "created_at", "category", "priority", "status"]].copy()
    latest["category"] = latest["category"].map(CATEGORY_LABELS)
    latest.columns = ["ID", "Date", "Category", "Priority", "Status"]
    st.dataframe(latest, hide_index=True)
    report_button(df, key="csv_dashboard")


# ----------------------------------------------------------------- page: insights
def page_insights() -> None:
    st.title("AI-Powered Insights")
    df = store.list_complaints()

    st.subheader("Key findings")
    for line in analytics.rule_based_insights(df):
        st.markdown(f"- {line}")

    st.subheader("Management summary")
    if df.empty:
        return
    if not pipeline.ai_enabled:
        st.info("Enable an LLM (API key or local Ollama) to generate an AI-written management summary.")
        return
    if st.button("Generate AI summary", type="primary"):
        try:
            with st.spinner("Writing summary..."):
                st.session_state.ai_summary = pipeline.summarize_insights(analytics.stats_as_text(df))
        except Exception:
            st.error("The AI service is unavailable right now. Please try again later.")
    if st.session_state.get("ai_summary"):
        st.markdown(st.session_state.ai_summary)


# ---------------------------------------------------------------------------- main
PAGES = {
    "Customer Portal": page_customer,
    "Support Console": page_support,
    "Analytics Dashboard": page_dashboard,
    "AI Insights": page_insights,
}

PAGES[render_sidebar()]()
