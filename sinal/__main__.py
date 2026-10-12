"""Command line.

    python -m sinal run data/interviews --out runs/demo --model qwen/qwen3-4b-2507
    python -m sinal run data/interviews --out runs/demo --offline     # replay from cache, no model
    python -m sinal eval runs/demo                                     # score against evals/golden.json
"""

import argparse
import json
import sys
from pathlib import Path

from . import evals, pipeline, report
from .llm import LLMError

DEFAULT_MODEL = "qwen/qwen3-4b-2507"
DEFAULT_GOLDEN = Path("evals/golden.json")


def write_report(run_dir: Path, golden: Path | None) -> dict | None:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    ev = None
    if golden and golden.exists():
        ev = evals.score(result, json.loads(golden.read_text(encoding="utf-8")))
        (run_dir / "evals.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "report.html").write_text(report.render(result, ev), encoding="utf-8")
    return ev


def print_evals(ev: dict) -> None:
    print(f"faithfulness     {ev['faithfulness']:.0%} of quotes literal  {ev['statuses']}")
    print(f"theme recall     {ev['theme_recall']:.0%}   precision {ev['theme_precision'] if ev['theme_precision'] is None else format(ev['theme_precision'], '.0%')}")
    print(f"contradictions   {ev['contradictions']}   trap hits {len(ev['trap_hits'])}   pii leaks {len(ev['pii_leaks'])}")
    print(f"model time       {ev['model_seconds']} s, {ev['tokens_per_second']} tokens/s")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sinal", description="Verifiable synthesis of user interviews with a local LLM.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyze a folder of transcripts")
    r.add_argument("interviews", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--offline", action="store_true", help="use cached model responses only")
    r.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)

    e = sub.add_parser("eval", help="score a finished run and rebuild its report")
    e.add_argument("run_dir", type=Path)
    e.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)

    args = p.parse_args(argv)
    try:
        if args.cmd == "run":
            pipeline.run(args.interviews, args.out, args.model, offline=args.offline)
            ev = write_report(args.out, args.golden)
        else:
            ev = write_report(args.run_dir, args.golden)
    except LLMError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if ev:
        print_evals(ev)
    print(f"report: {(args.out if args.cmd == 'run' else args.run_dir) / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
