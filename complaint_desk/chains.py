"""LangChain pipeline: classification -> priority + sentiment -> response.

Built with LangChain Expression Language (LCEL). Every step is a small chain
(prompt | llm | parser) and the steps are composed sequentially with
``RunnablePassthrough.assign`` so each stage can use the previous outputs.

    text ─▶ [Chain 1] category ─▶ [Chain 2] priority ┐
                                  [Chain 3] sentiment ┴▶ [Chain 4] response
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from . import fallback, prompts
from .config import CATEGORIES, PRIORITIES, SENTIMENTS, Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    category: str
    priority: str
    sentiment: str
    response: str
    mode: str  # "ai" or "offline"
    warning: str | None = None


def normalize_label(raw: str, allowed: Sequence[str]) -> str | None:
    """Map raw LLM output to one of the allowed labels (None if impossible)."""
    token = re.sub(r"[^a-z_ ]", "", raw.strip().lower()).replace(" ", "_")
    if token in allowed:
        return token
    for label in allowed:
        if label in token:
            return label
    return None


def _build_llm(settings: Settings) -> Runnable:
    from langchain_openai import ChatOpenAI  # imported lazily: offline mode needs no key

    return ChatOpenAI(
        model=settings.model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
        base_url=settings.base_url or None,
        timeout=30,
        max_retries=2,
    )


class ComplaintPipeline:
    """Runs the four chains, with a safe rule-based fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: Runnable | None = None,
        offline: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm: Runnable | None = None
        if not offline:
            if llm is not None:
                self.llm = llm
            elif self.settings.ai_enabled:
                self.llm = _build_llm(self.settings)

        self.chain: Runnable | None = None
        self.insights_chain: Runnable | None = None
        if self.llm is not None:
            self._build_chains(self.llm)

    # ------------------------------------------------------------------ setup
    @property
    def ai_enabled(self) -> bool:
        return self.llm is not None

    def _build_chains(self, llm: Runnable) -> None:
        parser = StrOutputParser()
        self.classify_chain = prompts.CLASSIFICATION_PROMPT | llm | parser
        self.priority_chain = prompts.PRIORITY_PROMPT | llm | parser
        self.sentiment_chain = prompts.SENTIMENT_PROMPT | llm | parser
        self.response_chain = prompts.RESPONSE_PROMPT | llm | parser
        self.insights_chain = prompts.INSIGHTS_PROMPT | llm | parser

        # Sequential composition: each stage adds a key to the running dict.
        self.chain = (
            RunnablePassthrough.assign(cat=RunnableLambda(self._classify))
            | RunnablePassthrough.assign(
                priority=RunnableLambda(self._priority),
                sentiment=RunnableLambda(self._sentiment),
            )
            | RunnablePassthrough.assign(response=self.response_chain)
        )

    # --------------------------------------------------------------- chain steps
    def _classify(self, inputs: dict[str, Any]) -> str:
        raw = self.classify_chain.invoke(inputs)
        label = normalize_label(raw, CATEGORIES)
        if label is None:
            logger.warning("Unparseable category %r - using rule-based fallback", raw)
            label = fallback.classify_category(inputs["text"])
        return label

    def _priority(self, inputs: dict[str, Any]) -> str:
        raw = self.priority_chain.invoke(inputs)
        label = normalize_label(raw, PRIORITIES)
        if label is None:
            logger.warning("Unparseable priority %r - using rule-based fallback", raw)
            label = fallback.assess_priority(inputs["text"], inputs["cat"])
        return label

    def _sentiment(self, inputs: dict[str, Any]) -> str:
        raw = self.sentiment_chain.invoke(inputs)
        label = normalize_label(raw, SENTIMENTS)
        if label is None:
            logger.warning("Unparseable sentiment %r - using rule-based fallback", raw)
            label = fallback.detect_sentiment(inputs["text"])
        return label

    # -------------------------------------------------------------------- public
    def analyze(self, text: str) -> AnalysisResult:
        """Classify a complaint and generate the customer acknowledgement."""
        if self.chain is None:
            return self._offline(text)
        try:
            out = self.chain.invoke({"text": text, "company": self.settings.company_name})
            return AnalysisResult(
                category=out["cat"],
                priority=out["priority"],
                sentiment=out["sentiment"],
                response=self._ensure_signature(out["response"].strip()),
                mode="ai",
            )
        except Exception:  # network errors, auth errors, rate limits, ...
            logger.exception("LLM pipeline failed")
            result = self._offline(text)
            result.warning = (
                "The AI service is temporarily unavailable, so your complaint was "
                "processed with our standard rules instead."
            )
            return result

    def summarize_insights(self, stats: str) -> str | None:
        """LLM-written management summary, or None in offline mode."""
        if self.insights_chain is None:
            return None
        return self.insights_chain.invoke({"stats": stats}).strip()

    # ------------------------------------------------------------------ helpers
    def _offline(self, text: str) -> AnalysisResult:
        category = fallback.classify_category(text)
        sentiment = fallback.detect_sentiment(text)
        return AnalysisResult(
            category=category,
            priority=fallback.assess_priority(text, category),
            sentiment=sentiment,
            response=fallback.build_response(category, sentiment, self.settings.company_name),
            mode="offline",
        )

    def _ensure_signature(self, response: str) -> str:
        company = self.settings.company_name
        if company.lower() not in response.lower():
            response = f"{response}\n\nRegards,\n{company}"
        return response
