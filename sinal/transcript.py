"""Read interview transcripts written as Markdown.

Format (one file per interview):

    ---
    id: P1
    perfil: Designer gráfica, 31 anos
    ---
    **Entrevistadora:** pergunta...
    **P1:** resposta...

Any speaker whose label equals the interview id is the participant; everyone else is the interviewer.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

TURN = re.compile(r"^\*\*(?P<speaker>[^*:]+):\*\*\s*(?P<text>.*)$")


@dataclass
class Turn:
    n: int            # 1-based turn number, shown to the model and in the report
    speaker: str
    is_participant: bool
    text: str


@dataclass
class Transcript:
    id: str
    profile: str
    turns: list[Turn] = field(default_factory=list)

    def participant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.is_participant]

    def turn(self, n: int) -> Turn | None:
        return next((t for t in self.turns if t.n == n), None)

    def numbered(self) -> str:
        """Transcript as the model sees it: one line per turn, tagged with its number."""
        return "\n".join(f"[{t.n}] {t.speaker}: {t.text}" for t in self.turns)


def parse_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    _, head, body = text.split("---", 2)
    meta = {}
    for line in head.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def parse(text: str) -> Transcript:
    meta, body = parse_front_matter(text)
    if "id" not in meta:
        raise ValueError("Transcript front matter must have an 'id'")
    tr = Transcript(id=meta["id"], profile=meta.get("perfil", ""))
    for line in body.splitlines():
        m = TURN.match(line.strip())
        if m:
            speaker = m["speaker"].strip()
            tr.turns.append(Turn(len(tr.turns) + 1, speaker, speaker == tr.id, m["text"].strip()))
        elif line.strip() and tr.turns:
            tr.turns[-1].text += " " + line.strip()  # wrapped paragraph continues the last turn
    if not tr.participant_turns():
        raise ValueError(f"No turns from participant {tr.id}")
    return tr


def load_dir(path: Path) -> list[Transcript]:
    files = sorted(Path(path).glob("*.md"))
    if not files:
        raise SystemExit(f"No .md transcripts in {path}")
    return [parse(f.read_text(encoding="utf-8")) for f in files]
