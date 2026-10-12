"""The whole run: read -> redact -> extract (model) -> verify (code) -> group (model) -> validate (code)."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import extract, themes, verify
from .llm import LocalLLM
from .redact import redact_transcript
from .transcript import load_dir


def run(interviews_dir: Path, run_dir: Path, model: str, offline: bool = False, log=print) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = LocalLLM(model, run_dir, offline=offline)

    transcripts = load_dir(interviews_dir)
    redactions = {tr.id: found for tr in transcripts if (found := redact_transcript(tr))}
    log(f"{len(transcripts)} interviews; personal data redacted in: {', '.join(redactions) or 'none'}")

    insights = []
    for tr in transcripts:
        raw = extract.extract(llm, tr)
        checked = verify.check_all(raw, tr)
        kept = sum(1 for i in checked if i["status"] in verify.KEEP)
        log(f"  {tr.id}: {len(raw)} insights, {kept} with a literal quote")
        insights += checked

    kept = [i for i in insights if i["status"] in verify.KEEP]
    grouped = themes.group(llm, kept)
    log(f"{len(kept)}/{len(insights)} insights kept; {sum(t['status'] == 'pattern' for t in grouped)} patterns")

    calls = []
    if llm.log_path.exists():
        calls = [json.loads(line) for line in llm.log_path.read_text(encoding="utf-8").splitlines()]
    result = {
        "meta": {
            "model": model,
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "interviews": len(transcripts),
            "pattern_min": themes.PATTERN_MIN,
        },
        "transcripts": [{"id": tr.id, "profile": tr.profile, "turns": [asdict(t) for t in tr.turns]}
                        for tr in transcripts],
        "redactions": redactions,
        "insights": insights,
        "themes": grouped,
        "calls": calls,
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
