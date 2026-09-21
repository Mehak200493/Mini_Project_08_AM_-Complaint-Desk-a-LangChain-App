"""Load realistic sample complaints so the dashboard looks alive.

Usage:
    python scripts/seed_demo_data.py            # add 45 sample complaints
    python scripts/seed_demo_data.py --reset    # wipe existing data first
    python scripts/seed_demo_data.py --count 80

Uses the offline rule-based engine, so it needs no API key and costs nothing.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_desk.chains import ComplaintPipeline  # noqa: E402
from complaint_desk.config import get_settings  # noqa: E402
from complaint_desk.storage import ComplaintStore  # noqa: E402

TEMPLATES = {
    "billing": [
        "My credit card was charged twice for the same purchase.",
        "An extra fee of ₹499 was deducted from my account without any notice.",
        "I still have not received my refund even after 10 days.",
        "Wrong transaction charge appeared on my monthly statement.",
        "This is unacceptable, a hidden fee was charged again and nobody replies to my emails.",
    ],
    "loan": [
        "My EMI payment was deducted twice this month.",
        "My loan approval is delayed and nobody is giving me an update.",
        "The interest rate on my loan was changed without informing me.",
        "EMI was debited even after I completed foreclosure of my loan.",
        "I have been waiting many days for my loan disbursement and there is still no response.",
    ],
    "fraud": [
        "There is an unauthorized transaction of ₹50,000 on my account.",
        "I think my account was hacked, I see suspicious activity and login alerts.",
        "Someone made a transaction of Rs. 12,000 that I did not make. Nobody is helping me!",
        "I received a phishing call and money is missing from my account.",
    ],
    "app_issue": [
        "The mobile app crashes every time I try to log in.",
        "My UPI payment failed with an error even though the app showed success.",
        "I cannot reset my password, the app keeps showing an error.",
        "The application freezes on the payment screen again and again.",
    ],
}
# Fraud is weighted towards the most recent week so the trend insight is interesting.
WEIGHTS = {"billing": 24, "loan": 26, "fraud": 20, "app_issue": 30}
NAMES = ["Aarav", "Diya", "Rohan", "Sneha", "Vikram", "Ananya", "Karthik", "Meera", "Arjun", "Isha", ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo complaints")
    parser.add_argument("--count", type=int, default=45)
    parser.add_argument("--reset", action="store_true", help="delete existing complaints first")
    args = parser.parse_args()

    settings = get_settings()
    store = ComplaintStore(settings.db_path)
    pipeline = ComplaintPipeline(settings, offline=True)
    if args.reset:
        store.clear()

    rng = random.Random(42)
    now = datetime.now()
    entries = []
    for _ in range(args.count):
        category = rng.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
        max_days = 6 if category == "fraud" and rng.random() < 0.6 else 20
        when = now - timedelta(days=rng.uniform(0, max_days), hours=rng.uniform(0, 5))
        entries.append((when, rng.choice(TEMPLATES[category])))
    entries.sort(key=lambda item: item[0])  # oldest first -> sequential IDs

    for when, text in entries:
        analysis = pipeline.analyze(text)
        age_days = (now - when).days
        if age_days > 10:
            status = rng.choices(["Closed", "In Progress", "Open"], weights=[80, 15, 5])[0]
        elif age_days > 3:
            status = rng.choices(["Closed", "In Progress", "Open"], weights=[45, 35, 20])[0]
        else:
            status = rng.choices(["Closed", "In Progress", "Open"], weights=[10, 30, 60])[0]
        store.add(
            complaint=text,
            category=analysis.category,
            priority=analysis.priority,
            sentiment=analysis.sentiment,
            response=analysis.response,
            customer_name=rng.choice(NAMES),
            mode="offline",
            status=status,
            created_at=when,
        )
    print(f"Added {len(entries)} sample complaints. Total in database: {store.count()}")


if __name__ == "__main__":
    main()
