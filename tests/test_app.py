"""End-to-end smoke test of the Streamlit UI (runs fully offline)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from complaint_desk.storage import ComplaintStore


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _app(monkeypatch, tmp_path) -> AppTest:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SUPPORT_PIN", "")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "ui.db"))
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_full_flow(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path)
    assert not at.exception

    # Submit a valid complaint through the chat box
    at.chat_input[0].set_value("My EMI payment was deducted twice from my account.").run()
    assert not at.exception
    store = ComplaintStore(tmp_path / "ui.db")
    assert store.count() == 1
    assert store.list_complaints().iloc[0]["category"] == "loan"

    # An invalid (too short) complaint is rejected and not stored
    at.chat_input[0].set_value("help").run()
    assert not at.exception
    assert store.count() == 1

    # Every other page renders without errors
    for page in ["Support Console", "Analytics Dashboard", "AI Insights"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
