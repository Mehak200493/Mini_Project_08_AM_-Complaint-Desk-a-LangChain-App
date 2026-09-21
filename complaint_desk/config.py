"""Central configuration: environment settings and domain constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Domain constants -------------------------------------------------------
# To add a new category (e.g. "kyc"), add it here, in CATEGORY_LABELS,
# and in the classification prompt / fallback keywords.
CATEGORIES = ["billing", "loan", "fraud", "app_issue"]
CATEGORY_LABELS = {
    "billing": "Billing",
    "loan": "Loan",
    "fraud": "Fraud",
    "app_issue": "App Issue",
}
PRIORITIES = ["high", "medium", "low"]
SENTIMENTS = ["angry", "frustrated", "neutral", "positive"]
STATUSES = ["Open", "In Progress", "Closed"]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model: str
    temperature: float
    company_name: str
    db_path: Path
    offline_mode: bool
    support_pin: str
    base_url: str = ""  # e.g. https://api.groq.com/openai/v1 for OpenAI-compatible providers
    llm_provider: str = "openai"  # "openai" (any OpenAI-compatible API) or "ollama" (local)
    ollama_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434"

    @property
    def ai_enabled(self) -> bool:
        """True when the LLM pipeline can be used (Ollama needs no API key)."""
        if self.offline_mode:
            return False
        return True if self.llm_provider == "ollama" else bool(self.openai_api_key)

    @property
    def active_model(self) -> str:
        """Name of the model that is actually used for the selected provider."""
        return self.ollama_model if self.llm_provider == "ollama" else self.model


def get_settings() -> Settings:
    """Read settings from the environment (and .env) on every call."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "ollama"}:
        provider = "openai"

    db_path = Path(os.getenv("DB_PATH", "data/complaints.db"))
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        # 0.3 -> consistent, professional, low-randomness answers for finance.
        temperature=_float(os.getenv("LLM_TEMPERATURE"), 0.3),
        company_name=os.getenv("COMPANY_NAME", "XYZ Finance").strip() or "XYZ Finance",
        db_path=db_path,
        offline_mode=_truthy(os.getenv("OFFLINE_MODE")),
        support_pin=os.getenv("SUPPORT_PIN", "").strip(),
        base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
        llm_provider=provider,
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral").strip() or "mistral",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        or "http://localhost:11434",
    )
