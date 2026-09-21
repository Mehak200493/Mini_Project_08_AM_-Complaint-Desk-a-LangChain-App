"""Input validation and sensitive-data masking for customer complaints."""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_LENGTH = 10
MAX_LENGTH = 1000

# 13-19 contiguous digits, or 4-4-4-x grouped digits (typical card numbers).
_CARD_RE = re.compile(r"\b(?:\d{4}[ -]){3}\d{1,7}\b|\b\d{13,19}\b")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    cleaned: str = ""
    error: str = ""


def mask_card_numbers(text: str) -> str:
    """Replace card-like numbers with XXXX-XXXX-XXXX-<last4>."""

    def _mask(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        if not 13 <= len(digits) <= 19:
            return match.group()
        return f"XXXX-XXXX-XXXX-{digits[-4:]}"

    return _CARD_RE.sub(_mask, text)


def validate_complaint(text: str | None) -> ValidationResult:
    """Validate raw complaint text and return a cleaned, masked version."""
    if text is None or not text.strip():
        return ValidationResult(False, error="Please describe your complaint before submitting.")

    cleaned = re.sub(r"[ \t]+", " ", text.strip())

    if len(cleaned) < MIN_LENGTH:
        return ValidationResult(
            False,
            error=f"Your complaint is too short. Please give at least {MIN_LENGTH} characters of detail.",
        )
    if len(cleaned) > MAX_LENGTH:
        return ValidationResult(
            False,
            error=f"Your complaint is too long ({len(cleaned)} characters). Please keep it under {MAX_LENGTH}.",
        )
    if sum(ch.isalpha() for ch in cleaned) < 5:
        return ValidationResult(False, error="Please describe the issue in words so we can help you.")

    return ValidationResult(True, cleaned=mask_card_numbers(cleaned))
