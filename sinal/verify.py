"""Check every quote the model returns against the transcript. No model involved.

A claim survives only if its quote is something the participant actually said. Status per insight:

- verified             quote found in the participant turn the model cited
- verified_other_turn  found in another participant turn (turn number corrected)
- interviewer_words    found only in what the interviewer said: the model put words in the participant's mouth
- paraphrased          close to a participant turn but not literal (similarity >= PARAPHRASE_MIN)
- not_found            nothing close in the transcript
- too_short            fewer than MIN_WORDS words; too short to support a claim
"""

import re
import unicodedata
from difflib import SequenceMatcher

MIN_WORDS = 4
PARAPHRASE_MIN = 0.8
KEEP = {"verified", "verified_other_turn"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation and quote marks never decide a match
    return re.sub(r"\s+", " ", text).strip()


def best_window_ratio(needle: str, hay: str) -> float:
    """Similarity of `needle` to its best-matching window of the same length in `hay` (word level)."""
    nw, hw = needle.split(), hay.split()
    if not nw or not hw:
        return 0.0
    size = len(nw)
    best = 0.0
    for i in range(max(1, len(hw) - size + 1)):
        window = " ".join(hw[i:i + size])
        best = max(best, SequenceMatcher(None, needle, window).ratio())
    return best


def check(insight: dict, tr) -> dict:
    """Return a copy of the insight with 'status', 'turn' (possibly corrected) and 'similarity'."""
    out = dict(insight)
    q = normalize(insight.get("quote", ""))
    if len(q.split()) < MIN_WORDS:
        out.update(status="too_short", similarity=0.0)
        return out

    cited = tr.turn(insight.get("turn", -1))
    if cited and cited.is_participant and q in normalize(cited.text):
        out.update(status="verified", similarity=1.0)
        return out
    for t in tr.participant_turns():
        if q in normalize(t.text):
            out.update(status="verified_other_turn", turn=t.n, similarity=1.0)
            return out
    for t in tr.turns:
        if not t.is_participant and q in normalize(t.text):
            out.update(status="interviewer_words", turn=t.n, similarity=1.0)
            return out

    scored = [(best_window_ratio(q, normalize(t.text)), t.n) for t in tr.participant_turns()]
    ratio, turn = max(scored)
    out.update(status="paraphrased" if ratio >= PARAPHRASE_MIN else "not_found",
               similarity=round(ratio, 3), nearest_turn=turn)
    return out


def check_all(insights: list[dict], tr) -> list[dict]:
    return [check(i, tr) for i in insights]
