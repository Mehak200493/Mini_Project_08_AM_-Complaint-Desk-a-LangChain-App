from datetime import datetime, timedelta

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from complaint_desk import analytics, fallback
from complaint_desk.chains import ComplaintPipeline, normalize_label
from complaint_desk.config import CATEGORIES, Settings
from complaint_desk.storage import ComplaintStore
from complaint_desk.validators import mask_card_numbers, validate_complaint


def make_settings(tmp_path, key="") -> Settings:
    return Settings(
        openai_api_key=key,
        model="gpt-4o-mini",
        temperature=0.3,
        company_name="XYZ Finance",
        db_path=tmp_path / "test.db",
        offline_mode=False,
        support_pin="",
    )


# ------------------------------------------------------------------ validators
def test_validation_rejects_empty_short_and_long():
    assert not validate_complaint("").ok
    assert not validate_complaint(None).ok
    assert not validate_complaint("help").ok
    assert not validate_complaint("x" * 1001).ok


def test_validation_accepts_and_masks_cards():
    result = validate_complaint("My card 4111 1111 1111 1234 was charged twice yesterday")
    assert result.ok
    assert "4111 1111" not in result.cleaned
    assert result.cleaned.count("1234") == 1
    assert mask_card_numbers("call 9876543210 amount 50000") == "call 9876543210 amount 50000"


# -------------------------------------------------------------------- fallback
@pytest.mark.parametrize(
    "text,expected",
    [
        ("My EMI payment was deducted twice.", "loan"),
        ("My credit card was charged twice.", "billing"),
        ("Unauthorized transaction of ₹50,000 on my account", "fraud"),
        ("The app crashes when I try to login", "app_issue"),
    ],
)
def test_fallback_category(text, expected):
    assert fallback.classify_category(text) == expected


def test_fallback_priority_and_sentiment():
    assert fallback.assess_priority("Unauthorized transaction of ₹50,000", "fraud") == "high"
    assert fallback.assess_priority("Refund not received", "billing") == "medium"
    assert fallback.assess_priority("App looks different", "app_issue") == "low"
    assert fallback.detect_sentiment("Nobody is helping me and money is missing.") == "angry"
    assert fallback.detect_sentiment("Thank you for the quick help") == "positive"
    assert fallback.extract_amount("charged Rs. 1,200 and ₹5,000") == 5000


def test_normalize_label():
    assert normalize_label(" Loan. ", CATEGORIES) == "loan"
    assert normalize_label("App Issue", CATEGORIES) == "app_issue"
    assert normalize_label("category: fraud", CATEGORIES) == "fraud"
    assert normalize_label("gibberish", CATEGORIES) is None


# --------------------------------------------------------------------- storage
def test_store_ids_filters_and_status(tmp_path):
    store = ComplaintStore(tmp_path / "s.db")
    common = dict(response="ok", sentiment="neutral")
    a = store.add(complaint="EMI deducted twice", category="loan", priority="medium", **common)
    b = store.add(complaint="Unauthorized txn", category="fraud", priority="high", **common)
    year = datetime.now().year
    assert a == f"CMP-{year}-001" and b == f"CMP-{year}-002"

    assert store.count() == 2
    assert len(store.list_complaints(categories=["fraud"])) == 1
    assert len(store.list_complaints(search="emi")) == 1
    assert store.get(a.lower())["category"] == "loan"

    assert store.update_status(a, "Closed")
    assert store.get(a)["status"] == "Closed"
    assert not store.update_status("CMP-0000-999", "Closed")
    with pytest.raises(ValueError):
        store.update_status(a, "Nonsense")


# ------------------------------------------------------------------- analytics
def test_analytics_metrics_and_insights(tmp_path):
    store = ComplaintStore(tmp_path / "a.db")
    now = datetime.now()
    for days_ago, cat, prio in [(1, "fraud", "high"), (2, "fraud", "high"), (9, "fraud", "high"), (3, "loan", "medium")]:
        store.add(
            complaint="sample complaint text",
            category=cat,
            priority=prio,
            sentiment="angry",
            response="ok",
            created_at=now - timedelta(days=days_ago),
        )
    df = store.list_complaints()
    m = analytics.summary_metrics(df)
    assert m["total"] == 4 and m["fraud"] == 3 and m["open"] == 4
    assert analytics.category_counts(df).set_index("Category").loc["Fraud", "Count"] == 3
    insights = " ".join(analytics.rule_based_insights(df, now=now))
    assert "Fraud" in insights and "increased by 100%" in insights
    assert "Fraud=3" in analytics.stats_as_text(df, now=now)
    assert not analytics.daily_trend(df).empty
    assert analytics.summary_metrics(store.list_complaints(search="zzz"))["total"] == 0


# -------------------------------------------------------------------- pipeline
def _fake_llm(prompt_value):
    text = prompt_value.to_string()
    if "complaint classifier" in text:
        return AIMessage(content="Loan.")
    if "triage officer" in text:
        return AIMessage(content="Medium")
    if "sentiment analyst" in text:
        return AIMessage(content="frustrated")
    if "customer support executive" in text:
        return AIMessage(content="Dear Customer,\n\nWe are investigating with our support team.")
    return AIMessage(content="summary")


def test_pipeline_with_fake_llm(tmp_path):
    pipe = ComplaintPipeline(make_settings(tmp_path), llm=RunnableLambda(_fake_llm))
    assert pipe.ai_enabled
    result = pipe.analyze("My EMI payment was deducted twice.")
    assert (result.category, result.priority, result.sentiment) == ("loan", "medium", "frustrated")
    assert result.mode == "ai" and result.warning is None
    assert result.response.rstrip().endswith("XYZ Finance")  # signature enforced
    assert pipe.summarize_insights("stats") == "summary"


def test_pipeline_falls_back_when_llm_fails(tmp_path):
    def boom(_):
        raise RuntimeError("API down")

    pipe = ComplaintPipeline(make_settings(tmp_path), llm=RunnableLambda(boom))
    result = pipe.analyze("Unauthorized transaction of ₹50,000 on my account")
    assert result.mode == "offline" and result.warning
    assert (result.category, result.priority) == ("fraud", "high")


def test_pipeline_offline_without_key(tmp_path):
    pipe = ComplaintPipeline(make_settings(tmp_path))
    assert not pipe.ai_enabled
    result = pipe.analyze("My credit card was charged twice.")
    assert result.category == "billing" and result.mode == "offline"
    assert "XYZ Finance" in result.response
    assert pipe.summarize_insights("x") is None


def test_settings_reads_base_url(monkeypatch):
    from complaint_desk.config import get_settings

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("OPENAI_MODEL", "llama-3.1-8b-instant")
    s = get_settings()
    assert s.base_url == "https://api.groq.com/openai/v1"
    assert s.model == "llama-3.1-8b-instant"
