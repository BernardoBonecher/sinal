"""Render a run as one self-contained HTML file (no JavaScript, no external assets)."""

from html import escape

STATUS_PT = {
    "verified": "Citação literal",
    "verified_other_turn": "Citação literal (outra fala)",
    "paraphrased": "Paráfrase, não literal",
    "not_found": "Não encontrada na entrevista",
    "interviewer_words": "Palavras do entrevistador",
    "too_short": "Citação curta demais",
}
KIND_PT = {"dor": "Dor", "comportamento": "Comportamento", "desejo": "Desejo"}


def pct(x) -> str:
    return "—" if x is None else f"{round(x * 100)}%"


def turn_context(tr: dict, n: int) -> str:
    """The interviewer turn before the quote plus the participant turn itself."""
    turns = {t["n"]: t for t in tr["turns"]}
    parts = []
    for k in (n - 1, n):
        t = turns.get(k)
        if t:
            cls = "ctx-p" if t["is_participant"] else "ctx-i"
            parts.append(f'<p class="{cls}"><b>{escape(t["speaker"])}</b> {escape(t["text"])}</p>')
    return "".join(parts)


def quote_block(ins: dict, tr: dict) -> str:
    return f"""
      <li class="quote">
        <blockquote>“{escape(ins['quote'])}”</blockquote>
        <div class="q-meta"><span class="pill">{escape(ins['participant'])}</span>
          <span>{escape(KIND_PT.get(ins['kind'], ins['kind']))} · fala {ins['turn']}</span>
          <span class="claim">{escape(ins['claim'])}</span></div>
        <details><summary>Ver na entrevista</summary>{turn_context(tr, ins['turn'])}</details>
      </li>"""


def theme_card(theme: dict, insights: dict, transcripts: dict, total: int) -> str:
    quotes = "".join(quote_block(insights[i], transcripts[insights[i]["participant"]]) for i in theme["insight_ids"])
    people = "".join(f'<span class="pill">{escape(p)}</span>' for p in theme["participants"])
    n = len(theme["participants"])
    return f"""
    <article class="theme">
      <header>
        <h3>{escape(theme['name'])}</h3>
        <div class="reach"><span class="bar"><i style="width:{round(n / total * 100)}%"></i></span>
          <span>{n} de {total} pessoas</span></div>
      </header>
      <p class="summary">{escape(theme['summary'])}</p>
      <div class="people">{people}</div>
      <ul class="quotes">{quotes}</ul>
    </article>"""


def rejected_rows(insights: list[dict], transcripts: dict) -> str:
    rows = []
    for i in insights:
        if i["status"] in ("verified", "verified_other_turn"):
            continue
        nearest = ""
        if i["status"] == "not_found":
            nearest = f'<div class="nearest">Nenhum trecho parecido em nenhuma fala de {escape(i["participant"])} (no máximo {round(i["similarity"] * 100)}% de semelhança).</div>'
        elif i.get("nearest_turn"):
            t = next((t for t in transcripts[i["participant"]]["turns"] if t["n"] == i["nearest_turn"]), None)
            if t:
                nearest = f'<div class="nearest">Mais próximo (fala {t["n"]}, {round(i["similarity"] * 100)}% parecido): {escape(t["text"])}</div>'
        elif i["status"] == "interviewer_words":
            t = next((t for t in transcripts[i["participant"]]["turns"] if t["n"] == i["turn"]), None)
            if t:
                nearest = f'<div class="nearest">Quem disse foi {escape(t["speaker"])} (fala {t["n"]}).</div>'
        rows.append(f"""
        <tr><td><span class="pill">{escape(i['participant'])}</span></td>
          <td>{escape(i['claim'])}<div class="model-quote">“{escape(i['quote'])}”</div>{nearest}</td>
          <td><span class="status s-{i['status']}">{escape(STATUS_PT.get(i['status'], i['status']))}</span></td></tr>""")
    if not rows:
        return '<p class="muted">Nenhum achado descartado nesta execução.</p>'
    return f"""<div class="table-wrap"><table class="rejected"><thead><tr><th>Pessoa</th><th>O que o modelo afirmou</th><th>Por que saiu</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>"""


def evals_section(ev: dict | None) -> str:
    if not ev:
        return ""
    matches = "".join(f"""
      <tr><td>{escape(m['label'])}</td><td>{', '.join(m['expected'])}</td>
        <td>{escape(m['matched_theme'] or '—')}</td><td>{', '.join(m['found']) or '—'}</td>
        <td>{'✓' if m['matched_theme'] else '✗'}{(' · contradiz: ' + ', '.join(m['contradictions'])) if m['contradictions'] else ''}</td></tr>"""
                      for m in ev["theme_matches"])
    used = {m["matched_theme"] for m in ev["theme_matches"] if m["matched_theme"]}
    extra = [t for t in ev.get("patterns", []) if t not in used]
    unmatched = (f'<p><b>Padrões sem tema plantado correspondente:</b> {escape(", ".join(extra))}.</p>' if extra else "")
    traps = "".join(f"<li><b>{escape(t['trap'])}</b> ({escape(t['insight'])}): {escape(t['claim'])} <span class='muted'>— {escape(t['why'])}</span></li>"
                    for t in ev["trap_hits"]) or "<li>Nenhuma.</li>"
    return f"""
  <section>
    <h2>Avaliação contra o gabarito</h2>
    <p class="muted">As entrevistas são sintéticas e foram escritas com temas plantados (evals/golden.json). Isso permite medir o modelo, não só confiar nele.</p>
    <div class="stats">
      <div><b>{pct(ev['faithfulness'])}</b><span>das citações do modelo eram literais</span></div>
      <div><b>{pct(ev['theme_recall'])}</b><span>dos temas plantados foram encontrados</span></div>
      <div><b>{pct(ev['theme_precision'])}</b><span>dos padrões relatados correspondem a um tema plantado</span></div>
      <div><b>{ev['contradictions']}</b><span>pessoas postas num tema que elas contradizem</span></div>
      <div><b>{len(ev['pii_leaks'])}</b><span>dados pessoais vazados no resultado</span></div>
    </div>
    <p class="muted small">Padrão que não corresponde a nenhum tema plantado não é necessariamente falso: pode ser um tema
      dividido em dois, ou um tema real que não foi plantado. Por isso eles aparecem na lista abaixo da tabela.</p>
    <div class="table-wrap"><table><thead><tr><th>Tema plantado</th><th>Quem fala dele</th><th>Padrão encontrado</th><th>Com quem</th><th></th></tr></thead><tbody>{matches}</tbody></table></div>
    {unmatched}
    <h3>Perguntas indutivas</h3>
    <p class="muted">Em P2 e P6 o entrevistador sugere uma resposta e a pessoa discorda. Achados que repetem a sugestão:</p>
    <ul class="traps">{traps}</ul>
    <p class="muted small">Modelo: {ev['model_seconds']} s no total, {ev['output_tokens']} tokens gerados ({ev['tokens_per_second'] or '—'} tokens/s).</p>
  </section>"""


def render(result: dict, ev: dict | None = None) -> str:
    meta = result["meta"]
    transcripts = {t["id"]: t for t in result["transcripts"]}
    insights = {i["id"]: i for i in result["insights"]}
    total_people = meta["interviews"]
    kept = [i for i in result["insights"] if i["status"] in ("verified", "verified_other_turn")]
    patterns = [t for t in result["themes"] if t["status"] == "pattern"]
    isolated = [t for t in result["themes"] if t["status"] == "isolated"]
    ungrouped = [t for t in result["themes"] if t["status"] == "ungrouped"]
    n_all = len(result["insights"])
    redacted = sum(sum(v.values()) for v in result["redactions"].values())

    pattern_html = "".join(theme_card(t, insights, transcripts, total_people) for t in patterns) \
        or '<p class="muted">Nenhum tema apareceu em pessoas suficientes.</p>'
    isolated_html = "".join(theme_card(t, insights, transcripts, total_people) for t in isolated)
    n_ungrouped = sum(len(t["insight_ids"]) for t in ungrouped)
    ungrouped_html = "".join(quote_block(insights[i], transcripts[insights[i]["participant"]])
                             for t in ungrouped for i in t["insight_ids"])

    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sinal — síntese de {total_people} entrevistas</title>
<style>{CSS}</style></head>
<body><main>
  <header class="top">
    <p class="eyebrow">Sinal · síntese verificável de entrevistas</p>
    <h1>O que {total_people} pessoas disseram, e onde disseram</h1>
    <p class="muted">Toda citação abaixo foi conferida, palavra por palavra, contra a transcrição. O que o modelo
      afirmou sem sustentação está na seção <a href="#erros">O que o modelo errou</a>.</p>
    <p class="muted small">Modelo local {escape(meta['model'])} · gerado em {escape(meta['created'])} ·
      padrão = tema presente em pelo menos {meta['pattern_min']} pessoas · {redacted} dado(s) pessoal(is) removido(s) antes da análise</p>
  </header>

  <div class="stats">
    <div><b>{n_all}</b><span>achados propostos pelo modelo</span></div>
    <div><b>{len(kept)}</b><span>com citação literal ({pct(len(kept) / n_all if n_all else None)})</span></div>
    <div><b>{len(patterns)}</b><span>padrões (≥ {meta['pattern_min']} pessoas)</span></div>
    <div><b>{len(isolated)}</b><span>sinais isolados (&lt; {meta['pattern_min']} pessoas)</span></div>
  </div>

  <section><h2>Padrões</h2>{pattern_html}</section>
  {f'<section><h2>Sinais isolados</h2><p class="muted">Aparecem em poucas pessoas. Podem importar, mas não são padrão.</p>{isolated_html}</section>' if isolated else ''}
  {f'<section><h2>Achados sem tema</h2><details class="theme"><summary>{n_ungrouped} achados verificados que o modelo não colocou em nenhum tema</summary><ul class="quotes">{ungrouped_html}</ul></details></section>' if ungrouped else ''}
  <section id="erros"><h2>O que o modelo errou</h2>
    <p class="muted">Achados descartados porque a citação não é fala literal da pessoa.</p>
    {rejected_rows(result['insights'], transcripts)}</section>
  {evals_section(ev)}
  <footer class="muted small">Entrevistas sintéticas, escritas para demonstrar a ferramenta. Pessoas, falas e contatos são fictícios.
    Código: <a href="https://github.com/BernardoBonecher/sinal">github.com/BernardoBonecher/sinal</a></footer>
</main></body></html>"""


CSS = """
:root{color-scheme:light;--bg:#f7f6f2;--card:#fffefb;--ink:#191917;--ink2:#55534d;--muted:#77756e;--line:#e4e2da;--accent:#1f6f5c;--accent-soft:#e3f0ec;--warn:#9a3412;--warn-soft:#fbe9df}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#121211;--card:#1b1b19;--ink:#efede7;--ink2:#c6c3b8;--muted:#98958b;--line:#2d2c29;--accent:#5cc2a4;--accent-soft:#1d3029;--warn:#f0a07a;--warn-soft:#3a2219}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}a{color:var(--accent)}
.top h1{font-size:clamp(1.7rem,4.5vw,2.5rem);line-height:1.15;margin:4px 0 12px}.eyebrow{margin:0;font-weight:600;color:var(--accent)}
.muted{color:var(--muted)}.small{font-size:.85rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}
.stats div{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.stats b{display:block;font-size:1.9rem;line-height:1.1}.stats span{color:var(--ink2);font-size:.9rem}
h2{margin:40px 0 8px;font-size:1.4rem}h3{margin:0;font-size:1.1rem}
.theme{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:14px 0}
.theme header{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center}
.reach{display:flex;align-items:center;gap:8px;color:var(--ink2);font-size:.9rem}
.bar{width:90px;height:8px;border-radius:99px;background:var(--line);overflow:hidden;display:inline-block}.bar i{display:block;height:100%;background:var(--accent);border-radius:99px}
.summary{margin:8px 0;color:var(--ink2)}.people{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px}
.pill{display:inline-block;font-size:.78rem;font-weight:600;padding:1px 8px;border-radius:99px;background:var(--accent-soft);color:var(--accent)}
.quotes{list-style:none;padding:0;margin:0}.quote{border-top:1px solid var(--line);padding:12px 0}
blockquote{margin:0 0 6px;font-size:1.02rem}.q-meta{display:flex;gap:8px;flex-wrap:wrap;align-items:center;color:var(--muted);font-size:.85rem}
.claim{color:var(--ink2)}details{margin-top:6px;font-size:.88rem}summary{cursor:pointer;color:var(--accent)}
.ctx-i,.ctx-p{margin:6px 0;padding:8px 10px;border-radius:8px}.ctx-i{color:var(--muted)}.ctx-p{background:var(--accent-soft)}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:12px 0;background:var(--card)}
th,td{text-align:left;vertical-align:top;padding:8px 10px;border-bottom:1px solid var(--line)}th{color:var(--ink2)}
.model-quote{color:var(--muted);margin-top:4px}.nearest{margin-top:4px;font-size:.85rem;color:var(--ink2)}
.status{font-size:.78rem;font-weight:600;padding:2px 8px;border-radius:6px;background:var(--warn-soft);color:var(--warn);white-space:nowrap}
.traps li{margin:4px 0}footer{margin-top:48px}
.table-wrap{overflow-x:auto}
@media (max-width:640px){.status{white-space:normal}th,td{padding:6px}}
"""
