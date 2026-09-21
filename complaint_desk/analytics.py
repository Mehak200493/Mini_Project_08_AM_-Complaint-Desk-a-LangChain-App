"""Dashboard analytics: metrics, aggregations, trends and rule-based insights."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .config import CATEGORIES, CATEGORY_LABELS, PRIORITIES, SENTIMENTS, STATUSES


def summary_metrics(df: pd.DataFrame) -> dict[str, int]:
    """Headline numbers. 'open' includes complaints that are In Progress."""
    if df.empty:
        return {"total": 0, "fraud": 0, "loan": 0, "billing": 0, "app_issue": 0, "open": 0, "closed": 0}
    return {
        "total": int(len(df)),
        "fraud": int((df["category"] == "fraud").sum()),
        "loan": int((df["category"] == "loan").sum()),
        "billing": int((df["category"] == "billing").sum()),
        "app_issue": int((df["category"] == "app_issue").sum()),
        "open": int((df["status"] != "Closed").sum()),
        "closed": int((df["status"] == "Closed").sum()),
    }


def _counts(df: pd.DataFrame, column: str, order: list[str], labels: dict[str, str] | None = None) -> pd.DataFrame:
    counts = df[column].value_counts() if not df.empty else pd.Series(dtype=int)
    labels = labels or {k: k.title() for k in order}
    return pd.DataFrame(
        {
            "Label": [labels.get(k, k) for k in order],
            "Count": [int(counts.get(k, 0)) for k in order],
        }
    )


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = _counts(df, "category", CATEGORIES, CATEGORY_LABELS)
    return out.rename(columns={"Label": "Category"})


def priority_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = _counts(df, "priority", PRIORITIES)
    return out.rename(columns={"Label": "Priority"})


def sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = _counts(df, "sentiment", SENTIMENTS)
    return out.rename(columns={"Label": "Sentiment"})


def status_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["status"].value_counts() if not df.empty else pd.Series(dtype=int)
    return pd.DataFrame({"Status": STATUSES, "Count": [int(counts.get(s, 0)) for s in STATUSES]})


def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Complaints per day (zero-filled), indexed by date."""
    if df.empty:
        return pd.DataFrame({"Complaints": []}, index=pd.DatetimeIndex([]))
    days = df["created_at"].dt.normalize()
    series = days.value_counts().sort_index()
    full = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full, fill_value=0).rename("Complaints").to_frame()


def _window_counts(df: pd.DataFrame, category: str, now: datetime) -> tuple[int, int]:
    """(last 7 days, previous 7 days) complaint counts for a category."""
    sub = df[df["category"] == category]
    last = int((sub["created_at"] > now - timedelta(days=7)).sum())
    prev = int(
        ((sub["created_at"] <= now - timedelta(days=7)) & (sub["created_at"] > now - timedelta(days=14))).sum()
    )
    return last, prev


def rule_based_insights(df: pd.DataFrame, now: datetime | None = None) -> list[str]:
    """Plain-language management insights computed from the data."""
    if df.empty:
        return ["No complaints recorded yet. Insights will appear once customers submit complaints."]

    now = now or datetime.now()
    total = len(df)
    insights: list[str] = []

    top = category_counts(df).sort_values("Count", ascending=False).iloc[0]
    insights.append(
        f"**{top['Category']}** is the largest complaint category: "
        f"{int(top['Count'])} cases ({top['Count'] / total:.0%} of all complaints)."
    )

    high_share = (df["priority"] == "high").mean()
    open_high = int(((df["priority"] == "high") & (df["status"] != "Closed")).sum())
    insights.append(
        f"{high_share:.0%} of complaints are high priority; **{open_high}** high-priority "
        "complaint(s) are still unresolved."
    )

    upset_share = df["sentiment"].isin(["angry", "frustrated"]).mean()
    insights.append(f"{upset_share:.0%} of customers sound angry or frustrated.")

    for category in CATEGORIES:
        last, prev = _window_counts(df, category, now)
        label = CATEGORY_LABELS[category]
        if prev > 0:
            change = (last - prev) / prev
            if change > 0:
                insights.append(f"{label} complaints **increased by {change:.0%}** vs the previous 7 days ({prev} → {last}).")
            elif change < 0:
                insights.append(f"{label} complaints **decreased by {abs(change):.0%}** vs the previous 7 days ({prev} → {last}).")
        elif last > 0:
            insights.append(f"{label}: {last} new complaint(s) this week (none in the previous 7 days).")

    return insights


def stats_as_text(df: pd.DataFrame, now: datetime | None = None) -> str:
    """Compact statistics string used as the LLM insight prompt input."""
    now = now or datetime.now()
    m = summary_metrics(df)
    lines = [
        f"Total complaints: {m['total']} (open/in progress: {m['open']}, closed: {m['closed']})",
        "By category: " + ", ".join(f"{r.Category}={r.Count}" for r in category_counts(df).itertuples()),
        "By priority: " + ", ".join(f"{r.Priority}={r.Count}" for r in priority_counts(df).itertuples()),
        "By sentiment: " + ", ".join(f"{r.Sentiment}={r.Count}" for r in sentiment_counts(df).itertuples()),
    ]
    if not df.empty:
        for category in CATEGORIES:
            last, prev = _window_counts(df, category, now)
            lines.append(f"{CATEGORY_LABELS[category]} last 7 days={last}, previous 7 days={prev}")
    return "\n".join(lines)
