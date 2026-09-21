"""Activity B - run the same 10 complaints through a hosted API and a local Ollama model.

Usage:
    python scripts/benchmark_llms.py                    # run both back-ends, write the report
    python scripts/benchmark_llms.py --only ollama      # run just one (results are saved per back-end)
    python scripts/benchmark_llms.py --report-only      # rebuild the report from saved results
    python scripts/benchmark_llms.py --hardware "Ryzen 5, 16 GB RAM, no GPU"

Before running:
    1. Hosted side : OPENAI_API_KEY (+ OPENAI_BASE_URL / OPENAI_MODEL) set in .env
    2. Local side  : install Ollama, then   ollama pull mistral
                     and   pip install langchain-ollama

Outputs (folder "reports/"):
    results_openai.json, results_ollama.json   raw measurements
    ACTIVITY_B_COMPARISON.md / .html           the one-page comparison (open the .html, Print -> PDF)
    ACTIVITY_B_DETAILS.md                      per-complaint labels and full replies
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_desk import benchmark  # noqa: E402
from complaint_desk.chains import build_llm  # noqa: E402
from complaint_desk.config import BASE_DIR, Settings, get_settings  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):  # avoid Windows console encoding crashes
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def hosted_label(settings: Settings) -> str:
    url = settings.base_url.lower()
    if "groq" in url:
        return "hosted on Groq"
    return "OpenAI API" if not url else f"hosted at {settings.base_url}"


def ollama_ready(settings: Settings) -> str | None:
    """Return None if Ollama is reachable and the model is pulled, else a helpful message."""
    base = settings.ollama_base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=4) as resp:
            names = [m.get("name", "") for m in json.load(resp).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return (f"Cannot reach Ollama at {base}. Install it from https://ollama.com, "
                "start it (on Windows it runs in the tray), then try again.")
    wanted = settings.ollama_model
    # "mistral" matches "mistral:latest"; "mistral:7b" must match exactly
    if not any(n == wanted or (":" not in wanted and n.split(":")[0] == wanted) for n in names):
        return f"Model '{wanted}' is not pulled yet. Run:  ollama pull {wanted}"
    return None


def save(result: benchmark.ProviderResult, out_dir: Path) -> None:
    path = out_dir / f"results_{result.key}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {path}")


def load(key: str, out_dir: Path) -> benchmark.ProviderResult | None:
    path = out_dir / f"results_{key}.json"
    if not path.exists():
        return None
    return benchmark.ProviderResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hosted API vs local Ollama benchmark")
    parser.add_argument("--only", choices=["openai", "ollama"], help="run just one back-end")
    parser.add_argument("--report-only", action="store_true", help="skip running; rebuild the report")
    parser.add_argument("--ollama-model", help="local model (default: OLLAMA_MODEL or 'mistral')")
    parser.add_argument("--hardware", default="", help='describe the local machine, e.g. "Ryzen 5, 16 GB, no GPU"')
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--price-in", type=float, default=0.075, help="hosted USD per 1M input tokens")
    parser.add_argument("--price-out", type=float, default=0.30, help="hosted USD per 1M output tokens")
    parser.add_argument("--server-monthly-usd", type=float, default=500.0,
                        help="ASSUMED monthly cost of a dedicated inference server")
    parser.add_argument("--out", default="reports", help="output folder (default: reports)")
    args = parser.parse_args()

    settings = get_settings()
    if args.ollama_model:
        settings = dataclasses.replace(settings, ollama_model=args.ollama_model)
    out_dir = (BASE_DIR / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        if args.only in (None, "openai"):
            print(f"\n[A] Hosted: {settings.model} ({hosted_label(settings)})")
            if not settings.openai_api_key:
                print("  skipped: OPENAI_API_KEY is not set in .env")
            else:
                result = benchmark.run_provider(
                    "openai", hosted_label(settings), settings.model,
                    build_llm(settings, "openai"), settings, warmup=not args.no_warmup,
                    endpoint=settings.base_url)
                save(result, out_dir)

        if args.only in (None, "ollama"):
            print(f"\n[B] Local: {settings.ollama_model} via Ollama")
            problem = ollama_ready(settings)
            if problem:
                print(f"  skipped: {problem}")
            else:
                result = benchmark.run_provider(
                    "ollama", "local, Ollama", settings.ollama_model,
                    build_llm(settings, "ollama"), settings, warmup=not args.no_warmup,
                    hardware=args.hardware)
                save(result, out_dir)

    a, b = load("openai", out_dir), load("ollama", out_dir)
    if not (a and b):
        missing = "hosted (openai)" if not a else "local (ollama)"
        print(f"\nNo comparison yet: results for the {missing} back-end are missing. "
              "Fix the message above, then run the script again.")
        return

    asm = benchmark.Assumptions(args.price_in, args.price_out, args.server_monthly_usd)
    (out_dir / "ACTIVITY_B_COMPARISON.md").write_text(benchmark.render_markdown(a, b, asm), encoding="utf-8")
    (out_dir / "ACTIVITY_B_COMPARISON.html").write_text(benchmark.render_html(a, b, asm), encoding="utf-8")
    (out_dir / "ACTIVITY_B_DETAILS.md").write_text(benchmark.render_details_markdown(a, b), encoding="utf-8")

    print("\n=== Summary ===")
    for r in (a, b):
        print(f"{r.model:<28} triage {r.triage_accuracy:5.1f}%  rubric {r.quality_score:5.1f}%  "
              f"median {r.median_latency:5.1f}s  failed {r.failed}")
    print(f"\nOpen and print to PDF: {out_dir / 'ACTIVITY_B_COMPARISON.html'}")


if __name__ == "__main__":
    main()
