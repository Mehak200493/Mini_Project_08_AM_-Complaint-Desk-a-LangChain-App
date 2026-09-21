"""Check that your LLM connection works (run this before starting the app).

Usage:
    python scripts/check_llm.py                # test a sample complaint with the active provider
    python scripts/check_llm.py --list-models  # show the models available to you

The provider is chosen by LLM_PROVIDER in .env:
    openai  (default) - any OpenAI-compatible API, e.g. Groq or OpenAI
    ollama            - a local model served by Ollama
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_desk.chains import ComplaintPipeline  # noqa: E402
from complaint_desk.config import get_settings  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):  # avoid Windows console encoding crashes
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def list_ollama_models(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=4) as resp:
        return sorted(m.get("name", "") for m in json.load(resp).get("models", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the LLM connection")
    parser.add_argument("--list-models", action="store_true", help="list models available to you")
    args = parser.parse_args()

    s = get_settings()
    local = s.llm_provider == "ollama"
    print(f"Provider : {s.llm_provider}" + (" (local)" if local else " (hosted, OpenAI-compatible)"))
    print(f"Model    : {s.active_model}")
    if local:
        print(f"Server   : {s.ollama_base_url}")
    else:
        print(f"Base URL : {s.base_url or 'https://api.openai.com/v1 (default)'}")
        print(f"API key  : {'set (' + s.openai_api_key[:4] + '...)' if s.openai_api_key else 'NOT SET'}")

    if not s.ai_enabled:
        print("\nNo API key found in .env -> the app will run in offline demo mode.")
        return

    if args.list_models:
        print("\nAvailable models:")
        try:
            if local:
                names = list_ollama_models(s.ollama_base_url)
            else:
                from openai import OpenAI

                client = OpenAI(api_key=s.openai_api_key, base_url=s.base_url or None)
                names = sorted(m.id for m in client.models.list().data)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  could not reach the server: {exc}\n  Is Ollama running? (ollama serve)")
            sys.exit(1)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            sys.exit(1)
        for name in names:
            print(f"  - {name}")
        if local and not names:
            print("  (none) - download one with:  ollama pull mistral")
        return

    pipeline = ComplaintPipeline(s)
    try:
        reply = pipeline.llm.invoke("Say hello in 5 words")  # surfaces the real error, if any
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}\n{exc}")
        if local:
            print(f"\nHints: is Ollama running?  Is the model pulled?  ->  ollama pull {s.ollama_model}")
        else:
            print("\nHints: 401 = wrong key | 404 = wrong/retired model name (try --list-models) | 429 = no credit / rate limit")
        sys.exit(1)

    print(f"\nLLM reply: {reply.content}")
    result = pipeline.analyze("My EMI payment was deducted twice.")
    print(f"Pipeline : mode={result.mode}, category={result.category}, "
          f"priority={result.priority}, sentiment={result.sentiment}")
    print("\nAll good - start the app with: streamlit run app.py" if result.mode == "ai"
          else "\nPipeline fell back to offline mode - check the logs above.")


if __name__ == "__main__":
    main()
