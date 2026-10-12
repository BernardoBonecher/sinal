"""Step 2 (model): group verified insights into themes. Step 3 (code): check the grouping.

The model only proposes groups of insight ids. Code decides what a theme is worth: ids that don't
exist are dropped, each theme's participants are counted from its ids, and only themes backed by at
least PATTERN_MIN different participants are called a pattern.
"""

import re

PATTERN_MIN = 3
MAX_THEMES = 10
MAX_IDS_PER_THEME = 20
# A 4B model under a JSON grammar can loop inside an unbounded array until the context runs out.
# Bounding every array and string, plus a token ceiling, makes that impossible.
MAX_TOKENS = 2000

SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "maxItems": MAX_THEMES,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 80},
                    "summary": {"type": "string", "maxLength": 300},
                    "insight_ids": {"type": "array", "maxItems": MAX_IDS_PER_THEME,
                                    "items": {"type": "string", "maxLength": 8}},
                },
                "required": ["name", "summary", "insight_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["themes"],
    "additionalProperties": False,
}

SYSTEM = """Você é um pesquisador de UX sintetizando várias entrevistas de discovery.
Agrupe os achados abaixo em temas.

Regras:
- Um tema junta achados que falam do MESMO problema ou comportamento, de pessoas diferentes.
- Use apenas os ids fornecidos. Cada achado entra em no máximo um tema. No máximo 10 temas.
- "name": até 8 palavras. "summary": uma ou duas frases, sem inventar nada além dos achados.
- Achado que não combina com nenhum outro pode ficar em um tema próprio.
- Responda em português."""


def messages_for(insights: list[dict]) -> list[dict]:
    lines = [f"{i['id']} ({i['participant']}, {i['kind']}): {i['claim']}" for i in insights]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Achados:\n" + "\n".join(lines)},
    ]


def group(llm, insights: list[dict]) -> list[dict]:
    if not insights:
        return []
    result = llm.json("themes", messages_for(insights), SCHEMA, label="themes", max_tokens=MAX_TOKENS)
    return validate(result.get("themes", []), insights)


ID_REFS = re.compile(r"\s*\((?:P\d+-\d+[,;\s]*)+\)")
DANGLING_REFS = re.compile(r"\s*\(\s*P\d[^)]*$")   # a ref list cut by the length limit


def clean_summary(text: str) -> str:
    """Drop id references from the prose and end on a complete sentence."""
    text = DANGLING_REFS.sub("", ID_REFS.sub("", text)).strip()
    if text and text[-1] not in ".!?":
        cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        text = text[:cut + 1] if cut > 0 else text + "…"
    return text


def validate(raw_themes: list[dict], insights: list[dict]) -> list[dict]:
    by_id = {i["id"]: i for i in insights}
    used: set[str] = set()
    themes = []
    for t in raw_themes:
        ids = [i for i in dict.fromkeys(t.get("insight_ids", [])) if i in by_id and i not in used]
        dropped = [i for i in t.get("insight_ids", []) if i not in by_id]
        if not ids:
            continue
        used.update(ids)
        participants = sorted({by_id[i]["participant"] for i in ids}, key=lambda p: (len(p), p))
        themes.append({
            "name": t.get("name", "").strip() or "Sem nome",
            # Summaries sometimes cite ids that aren't in the theme; the report shows the real
            # evidence, so strip id references from the prose.
            "summary": clean_summary(t.get("summary", "")),
            "insight_ids": ids,
            "participants": participants,
            "status": "pattern" if len(participants) >= PATTERN_MIN else "isolated",
            "unknown_ids": dropped,
        })
    themes.sort(key=lambda t: (t["status"] != "pattern", -len(t["participants"])))
    orphans = [i for i in by_id if i not in used]
    if orphans:
        themes.append({
            "name": "Achados sem tema", "summary": "O modelo não agrupou estes achados.",
            "insight_ids": orphans,
            "participants": sorted({by_id[i]["participant"] for i in orphans}, key=lambda p: (len(p), p)),
            "status": "ungrouped", "unknown_ids": [],
        })
    return themes
