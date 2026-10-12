"""Step 1 (model): pull insights, each with a literal quote, out of one interview."""

MAX_INSIGHTS = 8
MAX_TOKENS = 1500

SCHEMA = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "maxItems": MAX_INSIGHTS,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "kind": {"type": "string", "enum": ["dor", "comportamento", "desejo"]},
                    "quote": {"type": "string"},
                    "turn": {"type": "integer"},
                },
                "required": ["claim", "kind", "quote", "turn"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["insights"],
    "additionalProperties": False,
}

SYSTEM = """Você é um pesquisador de UX analisando uma entrevista de discovery.
Extraia os achados mais importantes sobre a pessoa entrevistada ({pid}).

Regras:
- Cada achado tem: "claim" (uma frase curta, sua síntese), "kind" (dor, comportamento ou desejo),
  "quote" (um trecho COPIADO LITERALMENTE de uma fala de {pid}, de 6 a 30 palavras) e
  "turn" (o número entre colchetes da fala de onde saiu a citação).
- A citação tem que ser fala de {pid}, nunca do entrevistador.
- Se o entrevistador sugerir algo e {pid} discordar, registre o que {pid} disse, não a sugestão.
- No máximo {max_insights} achados. Prefira poucos e bem sustentados.
- Responda em português."""


def messages_for(tr) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM.format(pid=tr.id, max_insights=MAX_INSIGHTS)},
        {"role": "user", "content": f"Perfil: {tr.profile}\n\nEntrevista:\n{tr.numbered()}"},
    ]


def extract(llm, tr) -> list[dict]:
    result = llm.json("insights", messages_for(tr), SCHEMA, label=f"extract {tr.id}", max_tokens=MAX_TOKENS)
    insights = result.get("insights", [])[:MAX_INSIGHTS]
    for i, ins in enumerate(insights, 1):
        ins["id"] = f"{tr.id}-{i}"
        ins["participant"] = tr.id
    return insights
