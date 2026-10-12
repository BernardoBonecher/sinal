"""Remove personal data before text reaches the model or the report.

Deterministic and conservative: e-mails, Brazilian phone numbers, CPF and CNPJ. Names are not
handled here; transcripts identify participants by code (P1, P2...) by convention.
"""

import re

PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")),
    ("cnpj", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("cpf", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("phone", re.compile(r"(\+?55\s?)?\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b")),
]
LABELS = {"email": "[e-mail]", "cnpj": "[CNPJ]", "cpf": "[CPF]", "phone": "[telefone]"}


def redact(text: str) -> tuple[str, dict[str, int]]:
    found: dict[str, int] = {}
    for kind, pattern in PATTERNS:
        text, n = pattern.subn(LABELS[kind], text)
        if n:
            found[kind] = found.get(kind, 0) + n
    return text, found


def redact_transcript(tr) -> dict[str, int]:
    """Redact every turn in place; return counts by kind."""
    total: dict[str, int] = {}
    for turn in tr.turns:
        turn.text, found = redact(turn.text)
        for k, v in found.items():
            total[k] = total.get(k, 0) + v
    return total
