"""Check that your LLM connection works (run this before starting the app).

Usage:
    python scripts/check_llm.py                # test a sample complaint
    python scripts/check_llm.py --list-models  # show the models your key can use
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_desk.chains import ComplaintPipeline  # noqa: E402
from complaint_desk.config import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the LLM connection")
    parser.add_argument("--list-models", action="store_true", help="list models available to your key")
    args = parser.parse_args()

    s = get_settings()
    print(f"Model    : {s.model}")
    print(f"Base URL : {s.base_url or 'https://api.openai.com/v1 (default)'}")
    print(f"API key  : {'set (' + s.openai_api_key[:4] + '...)' if s.openai_api_key else 'NOT SET'}")

    if not s.openai_api_key:
        print("\nNo API key found in .env -> the app will run in offline demo mode.")
        return

    if args.list_models:
        from openai import OpenAI

        client = OpenAI(api_key=s.openai_api_key, base_url=s.base_url or None)
        print("\nAvailable models:")
        for model in sorted(client.models.list().data, key=lambda m: m.id):
            print(f"  - {model.id}")
        return

    pipeline = ComplaintPipeline(s)
    try:
        reply = pipeline.llm.invoke("Say hello in 5 words")  # surfaces the real error, if any
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}\n{exc}")
        print("\nHints: 401 = wrong key | 404 = wrong model name | 429 = no credit / rate limit")
        sys.exit(1)

    print(f"\nLLM reply: {reply.content}")
    result = pipeline.analyze("My EMI payment was deducted twice.")
    print(f"Pipeline : mode={result.mode}, category={result.category}, "
          f"priority={result.priority}, sentiment={result.sentiment}")
    print("\nAll good - start the app with: streamlit run app.py" if result.mode == "ai"
          else "\nPipeline fell back to offline mode - check the logs above.")


if __name__ == "__main__":
    main()
