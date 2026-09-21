"""Rule-based fallback used when the LLM is unavailable.

It keeps the portal working for demos (no API key) and during OpenAI outages,
so a complaint is never lost. The UI always tells the user when it is used.
"""

from __future__ import annotations

import re

# Category keyword sets with weights (fraud/loan cues are stronger signals).
_CATEGORY_KEYWORDS: dict[str, tuple[float, list[str]]] = {
    "fraud": (
        3.0,
        [
            "unauthorized", "unauthorised", "fraud", "fraudulent", "hacked", "hack",
            "phishing", "scam", "stolen", "suspicious", "compromised", "not me",
            "did not make", "didn't make", "did not authorize", "not made by me",
        ],
    ),
    "loan": (
        2.5,
        [
            "emi", "loan", "interest rate", "tenure", "disbursal", "disbursement",
            "foreclosure", "sanction", "principal", "approval",
        ],
    ),
    "billing": (
        1.0,
        [
            "charged", "charge", "fee", "fees", "refund", "billing", "bill",
            "debited", "deducted", "invoice", "overcharged", "double", "twice",
            "cashback", "statement",
        ],
    ),
    "app_issue": (
        1.0,
        [
            "app", "application", "login", "log in", "crash", "crashes", "crashing",
            "password", "update", "error", "not opening", "freeze", "hang",
            "payment failed", "transaction failed", "website", "server",
        ],
    ),
}
_TIE_ORDER = ["fraud", "loan", "billing", "app_issue"]

_HIGH_PRIORITY_CUES = [
    "unauthorized", "unauthorised", "hacked", "stolen", "fraud", "scam",
    "money is missing", "money missing", "lost money", "phishing",
]

_ANGRY = [
    "worst", "useless", "furious", "angry", "cheat", "unacceptable", "pathetic",
    "ridiculous", "disgusting", "terrible", "outraged", "sick of", "nobody",
    "no one is helping", "no one helped", "!!!",
]
_FRUSTRATED = [
    "again", "still", "waiting", "not working", "multiple times", "many times",
    "no response", "tired", "frustrat", "annoy", "disappointed", "days",
]
_POSITIVE = ["thank", "great", "appreciate", "helpful", "happy"]

_AMOUNT_RE = re.compile(
    r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*(?:rupees|rs\b)",
    re.IGNORECASE,
)


def _contains(text: str, keyword: str) -> bool:
    if len(keyword) <= 4 and keyword.isalpha():
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
    return keyword in text


def classify_category(text: str) -> str:
    t = text.lower()
    scores: dict[str, float] = {}
    for category, (weight, words) in _CATEGORY_KEYWORDS.items():
        scores[category] = weight * sum(1 for w in words if _contains(t, w))
    best = max(_TIE_ORDER, key=lambda c: (scores[c], -_TIE_ORDER.index(c)))
    return best if scores[best] > 0 else "app_issue"


def extract_amount(text: str) -> float:
    """Largest rupee amount mentioned in the text (0 if none)."""
    amounts: list[float] = []
    for match in _AMOUNT_RE.finditer(text):
        raw = (match.group(1) or match.group(2) or "").replace(",", "")
        try:
            amounts.append(float(raw))
        except ValueError:
            continue
    return max(amounts, default=0.0)


def assess_priority(text: str, category: str) -> str:
    t = text.lower()
    amount = extract_amount(text)
    if category == "fraud" or any(cue in t for cue in _HIGH_PRIORITY_CUES) or amount >= 25000:
        return "high"
    if category in {"billing", "loan"} or amount >= 5000:
        return "medium"
    return "low"


def detect_sentiment(text: str) -> str:
    t = text.lower()
    if any(w in t for w in _ANGRY):
        return "angry"
    if any(w in t for w in _FRUSTRATED):
        return "frustrated"
    if any(w in t for w in _POSITIVE):
        return "positive"
    return "neutral"


_TOPICS = {
    "billing": "an incorrect charge or billing matter",
    "loan": "your loan or EMI account",
    "fraud": "suspicious activity on your account",
    "app_issue": "the issue you faced with our mobile app",
}


def build_response(category: str, sentiment: str, company: str) -> str:
    opening = (
        "We sincerely apologise for the inconvenience caused. "
        if sentiment in {"angry", "frustrated"}
        else ""
    )
    team = (
        "Our fraud investigation team has been notified and is reviewing your case."
        if category == "fraud"
        else "Our support team has started an investigation and will review the details carefully."
    )
    return (
        "Dear Customer,\n\n"
        f"{opening}Thank you for contacting {company}. We have received your complaint "
        f"regarding {_TOPICS.get(category, 'your account')}. {team} "
        "We will update you shortly and appreciate your patience and cooperation.\n\n"
        f"Regards,\n{company}"
    )
