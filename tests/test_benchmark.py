"""Tests for the Activity B benchmark harness (uses fake chat models - no network)."""

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from complaint_desk import benchmark, fallback
from complaint_desk.chains import build_llm
from complaint_desk.config import Settings, get_settings

REPLY = (
    "Dear Customer, thank you for contacting XYZ Finance. We understand your concern about this matter "
    "and sincerely apologise for the inconvenience. Our support team has started an investigation into "
    "your complaint and will review every detail carefully. We appreciate your patience while we look "
    "into it and will keep you informed. Regards, XYZ Finance"
)


class FakeChat(BaseChatModel):
    """Answers each chain from the fallback rules and reports token usage."""

    fail: bool = False
    reply: str = REPLY

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail:
            raise RuntimeError("model unavailable")
        system, human = messages[0].content, messages[-1].content
        text = human.split("Complaint:\n", 1)[-1]
        if "complaint classifier" in system:
            answer = fallback.classify_category(text)
        elif "triage officer" in system:
            answer = fallback.assess_priority(text, fallback.classify_category(text))
        elif "sentiment analyst" in system:
            answer = fallback.detect_sentiment(text)
        else:
            answer = self.reply
        msg = AIMessage(content=answer,
                        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_settings(tmp_path) -> Settings:
    return Settings(openai_api_key="x", model="fake-hosted", temperature=0.3, company_name="XYZ Finance",
                    db_path=tmp_path / "b.db", offline_mode=False, support_pin="")


def run(tmp_path, key, **fake_kwargs):
    return benchmark.run_provider(key, "test", f"fake-{key}", FakeChat(**fake_kwargs),
                                  make_settings(tmp_path), progress=lambda _: None)


# ---------------------------------------------------------------------------- scoring
def test_test_set_is_valid():
    assert len(benchmark.TEST_SET) == 10
    assert {t["category"] for t in benchmark.TEST_SET} == {"billing", "loan", "fraud", "app_issue"}
    assert [t["id"] for t in benchmark.TEST_SET] == list(range(1, 11))


def test_reply_word_count_is_in_spec():
    assert 50 <= benchmark.word_count(REPLY) <= 60


def test_quality_checks_pass_and_fail():
    good = benchmark.quality_checks(REPLY, "XYZ Finance", "loan")
    assert all(good.values()) and "fraud_team" not in good
    assert "fraud_team" in benchmark.quality_checks(REPLY, "XYZ Finance", "fraud")

    bad = "Dear Customer, we guarantee your refund within 2 days. Please share your OTP."
    checks = benchmark.quality_checks(bad, "XYZ Finance", "billing")
    assert not checks["no_promise"] and not checks["safe"]
    assert not checks["signature"] and not checks["length_50_60"]
    assert not benchmark.quality_checks("Card 4111 1111 1111 1234 noted. XYZ Finance", "XYZ Finance", "billing")["safe"]


def test_cost_math():
    assert benchmark.cost_per_1000(1000, 500, 0.10, 0.50) == pytest.approx(0.35)
    assert benchmark.breakeven_requests_per_month(500, 0.25) == pytest.approx(2_000_000)
    assert benchmark.breakeven_requests_per_month(500, 0) == float("inf")


# ---------------------------------------------------------------------------- running
def test_run_provider_collects_metrics(tmp_path):
    result = run(tmp_path, "openai")
    assert result.n == 10 and result.failed == 0
    rec = result.records[0]
    assert rec.input_tokens == 400 and rec.output_tokens == 40  # 4 LLM calls x (100 in, 10 out)
    assert not rec.tokens_estimated and rec.latency_s >= 0
    assert result.correct("category") >= 8  # rule-based fake gets the easy cases right
    # the canned reply never mentions fraud, so the 3 fraud complaints miss one of their 7 checks
    assert result.quality_score == pytest.approx(100 * (7 + 3 * 6 / 7) / 10)
    assert result.clean_outputs == 10
    assert result.median_latency >= 0 and result.avg_input_tokens == 400


def test_run_provider_records_failures(tmp_path):
    result = run(tmp_path, "ollama", fail=True)
    assert result.failed == 10 and result.n == 10
    assert result.triage_accuracy == 0 and "RuntimeError" in result.records[0].error


def test_raw_output_is_scored_without_auto_signature(tmp_path):
    result = run(tmp_path, "openai", reply="Dear Customer, we are investigating with our support team.")
    assert not any(r.checks["signature"] for r in result.ok_records)


def test_format_repairs_are_counted(tmp_path):
    class Chatty(FakeChat):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if "complaint classifier" in messages[0].content:
                msg = AIMessage(content="I am not sure", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
                return ChatResult(generations=[ChatGeneration(message=msg)])
            return super()._generate(messages, stop, run_manager, **kwargs)

    result = benchmark.run_provider("ollama", "t", "chatty", Chatty(), make_settings(tmp_path), progress=lambda _: None)
    assert result.clean_outputs == 0 and result.records[0].format_repairs == 1


# --------------------------------------------------------------------------- reporting
def test_reports_render_and_roundtrip(tmp_path):
    a, b = run(tmp_path, "openai"), run(tmp_path, "ollama")
    b.hardware = "test laptop"
    asm = benchmark.Assumptions()
    md, html = benchmark.render_markdown(a, b, asm), benchmark.render_html(a, b, asm)
    for text in (md, html):
        assert "Category correct" in text and "Cost per 1,000 requests" in text
        assert "Data privacy" in text or "Data privacy".upper() in text.upper()
    assert "ChatOllama" in md and "test laptop" in md
    assert "Which would you ship for a bank, and why?" in md and "print" in html.lower()
    assert len(benchmark.conclusion(a, b, asm)) == 3
    assert "Full replies" in benchmark.render_details_markdown(a, b)

    again = benchmark.ProviderResult.from_dict(json.loads(json.dumps(a.to_dict())))
    assert again.n == 10 and again.triage_accuracy == a.triage_accuracy


def test_conclusion_changes_with_accuracy_gap(tmp_path):
    a, b = run(tmp_path, "openai"), run(tmp_path, "ollama")
    asm = benchmark.Assumptions()
    assert "matched the hosted model" in benchmark.conclusion(a, b, asm)[2]
    b.records = [benchmark.Record(r.complaint_id, False, 1.0, error="x") for r in b.records]
    assert "trailed the hosted model" in benchmark.conclusion(a, b, asm)[2]


# ------------------------------------------------------------------------ provider switch
def test_provider_switch_builds_ollama_and_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    s = get_settings()
    assert s.llm_provider == "ollama" and s.active_model == "mistral" and s.ai_enabled
    llm = build_llm(s)
    assert type(llm).__name__ == "ChatOllama" and llm.model == "mistral" and llm.temperature == 0.3

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-oss-20b")
    s2 = get_settings()
    assert s2.active_model == "openai/gpt-oss-20b" and type(build_llm(s2)).__name__ == "ChatOpenAI"

    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    assert get_settings().llm_provider == "openai"


def test_speed_insight_names_the_faster_backend(tmp_path):
    a, b = run(tmp_path, "openai"), run(tmp_path, "ollama")
    for r in a.records:
        r.latency_s = 1.0
    for r in b.records:
        r.latency_s = 6.0
    asm = benchmark.Assumptions()
    assert "hosted back-end was 6.0x faster" in benchmark._insights(a, b)[0]
    assert "6.0x slower" in benchmark.conclusion(a, b, asm)[2]
    a.records, b.records = b.records, a.records  # now local is the faster one
    assert "local back-end was 6.0x faster" in benchmark._insights(a, b)[0]
    assert "6.0x faster" in benchmark.conclusion(a, b, asm)[2]
