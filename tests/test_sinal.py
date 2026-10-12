import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sinal import evals, report, themes, verify
from sinal.llm import LLMError, LocalLLM
from sinal.redact import LABELS, redact, redact_transcript
from sinal.transcript import parse

TRANSCRIPT = """---
id: P9
perfil: Teste
---

**Entrevistadora:** Você gostaria de um app que cobrasse automaticamente?

**P9:** Não sei, já baixei dois desses e larguei os dois porque cobravam taxa.

**Entrevistadora:** E como você controla?

**P9:** Numa planilha que uma amiga me passou,
com uma linha por aluno.
"""


@pytest.fixture
def tr():
    return parse(TRANSCRIPT)


# ---------- transcript ----------

def test_parse_roles_numbers_and_wrapped_lines(tr):
    assert tr.id == "P9" and tr.profile == "Teste"
    assert [t.n for t in tr.turns] == [1, 2, 3, 4]
    assert [t.is_participant for t in tr.turns] == [False, True, False, True]
    assert tr.turn(4).text == "Numa planilha que uma amiga me passou, com uma linha por aluno."
    assert "[2] P9: Não sei" in tr.numbered()


def test_parse_requires_participant_turns():
    with pytest.raises(ValueError):
        parse("---\nid: P1\n---\n**Entrevistadora:** Oi?\n")


# ---------- redaction ----------

@pytest.mark.parametrize("text, kind", [
    ("me escreve em ana.silva@exemplo.com.br", "email"),
    ("WhatsApp (51) 98877-1234", "phone"),
    ("liga 11 3456-7890", "phone"),
    ("CPF 123.456.789-09", "cpf"),
    ("CNPJ 12.345.678/0001-90", "cnpj"),
])
def test_redact_finds_personal_data(text, kind):
    out, found = redact(text)
    assert found == {kind: 1}
    assert LABELS[kind] in out
    assert not any(ch.isdigit() for ch in out)


def test_redact_leaves_money_and_years_alone():
    text = "Cobrei R$ 800 em 2026, quinze dias depois, uns 4 por cento."
    assert redact(text) == (text, {})


def test_redact_transcript_in_place():
    t = parse("---\nid: P1\n---\n**P1:** meu e-mail é a@b.com e o fone (48) 99812-0000\n")
    assert redact_transcript(t) == {"email": 1, "phone": 1}
    assert "@" not in t.turns[0].text and "99812" not in t.turns[0].text


# ---------- verification ----------

def ins(quote, turn=2, participant="P9"):
    return {"id": "P9-1", "participant": participant, "claim": "c", "kind": "dor", "quote": quote, "turn": turn}


def test_verified_ignores_case_and_punctuation(tr):
    out = verify.check(ins("já baixei dois desses e larguei os dois"), tr)
    assert out["status"] == "verified"


def test_verified_in_other_turn_corrects_turn(tr):
    out = verify.check(ins("Numa planilha que uma amiga me passou", turn=2), tr)
    assert out["status"] == "verified_other_turn" and out["turn"] == 4


def test_interviewer_words_are_caught(tr):
    out = verify.check(ins("gostaria de um app que cobrasse automaticamente", turn=2), tr)
    assert out["status"] == "interviewer_words" and out["turn"] == 1


def test_paraphrase_is_not_accepted(tr):
    out = verify.check(ins("já baixei dois desses e abandonei os dois"), tr)
    assert out["status"] == "paraphrased" and out["nearest_turn"] == 2


def test_invented_quote_not_found(tr):
    assert verify.check(ins("o maior problema é o imposto todo mês"), tr)["status"] == "not_found"


def test_short_quotes_rejected(tr):
    assert verify.check(ins("Não sei"), tr)["status"] == "too_short"


# ---------- theme validation ----------

def make_insights():
    return [{"id": f"{p}-1", "participant": p, "claim": "x", "kind": "dor", "quote": "q", "turn": 2}
            for p in ["P1", "P2", "P3", "P4"]]


def test_validate_counts_participants_and_marks_patterns():
    raw = [{"name": "A", "summary": "s", "insight_ids": ["P1-1", "P2-1", "P3-1"]},
           {"name": "B", "summary": "s", "insight_ids": ["P4-1"]}]
    out = themes.validate(raw, make_insights())
    assert [(t["name"], t["status"], t["participants"]) for t in out] == [
        ("A", "pattern", ["P1", "P2", "P3"]), ("B", "isolated", ["P4"])]


def test_validate_drops_unknown_and_reused_ids_and_collects_orphans():
    raw = [{"name": "A", "summary": "s", "insight_ids": ["P1-1", "P9-9", "P1-1"]},
           {"name": "B", "summary": "s", "insight_ids": ["P1-1", "P2-1"]}]
    out = themes.validate(raw, make_insights())
    a, b, orphans = out[0], out[1], out[-1]
    assert a["insight_ids"] == ["P1-1"] and a["unknown_ids"] == ["P9-9"]
    assert b["insight_ids"] == ["P2-1"]
    assert orphans["status"] == "ungrouped" and orphans["insight_ids"] == ["P3-1", "P4-1"]


# ---------- evals ----------

GOLDEN = {
    "themes": [
        {"id": "g1", "label": "G1", "participants": ["P1", "P2", "P3", "P4"], "counter": ["P5"], "keywords": ["atras"]},
        {"id": "g2", "label": "G2", "participants": ["P6", "P7", "P8"], "counter": [], "keywords": ["nota"]},
    ],
    "traps": [{"id": "t", "participant": "P6", "suspect_if": {"kind": ["desejo"], "claim_contains": ["app"]}}],
    "pii": [{"participant": "P2", "kind": "email", "value": "x@exemplo.com"}],
}


def theme(name, people, status="pattern", summary=""):
    return {"name": name, "summary": summary, "status": status, "participants": people,
            "insight_ids": [f"{p}-t" for p in people]}   # no insight text: judged by name


def insight(pid, claim="x", kind="dor", status="verified"):
    return {"id": f"{pid}-1", "participant": pid, "claim": claim, "kind": kind, "quote": "q", "status": status}


def result_with(themes_, insights_):
    return {"themes": themes_, "insights": insights_, "calls": [{"seconds": 10, "completion_tokens": 20}]}


def test_score_recall_precision_and_contradictions():
    th = [theme("Atraso no pagamento", ["P1", "P2", "P3", "P5"]),
          theme("Nota fiscal confusa", ["P1", "P6", "P8"]),
          theme("Solo", ["P7"], status="isolated")]
    insights_ = [insight("P6", claim="Quer um app de cobrança", kind="desejo"),
                 insight("P1", status="not_found")]
    ev = evals.score(result_with(th, insights_), GOLDEN)
    assert ev["theme_recall"] == 1.0
    assert ev["theme_precision"] == 1.0
    assert ev["contradictions"] == 1                # P5 contradicts g1 but sits in the matched theme
    assert ev["faithfulness"] == 0.5
    assert [h["insight"] for h in ev["trap_hits"]] == ["P6-1"]
    assert ev["pii_leaks"] == [] and ev["tokens_per_second"] == 2.0


def test_same_people_but_off_topic_is_not_a_match():
    # The bug this guards against: a theme about invoices matched a planted theme about late
    # payment only because the same people appear in both.
    th = [theme("Nota fiscal confusa", ["P1", "P2", "P3"])]
    ev = evals.score(result_with(th, []), GOLDEN)
    assert ev["theme_matches"][0]["matched_theme"] is None


def test_one_on_topic_insight_is_not_enough():
    # A theme mostly about spreadsheets with a single insight that mentions the planted subject.
    th = [theme("Controle manual", ["P1", "P2", "P3"])]
    ins_ = [insight("P1", claim="usa planilha"), insight("P2", claim="usa caderno"),
            insight("P3", claim="cliente atrasa o pagamento")]
    ev = evals.score(result_with(th, ins_), GOLDEN)
    assert ev["theme_matches"][0]["matched_theme"] is None


def test_larger_matching_theme_wins_over_subset():
    th = [theme("Atraso e desgaste", ["P1", "P2"]), theme("Atraso no pagamento", ["P1", "P2", "P3", "P4"])]
    ev = evals.score(result_with(th, []), GOLDEN)
    assert ev["theme_matches"][0]["matched_theme"] == "Atraso no pagamento"


def test_one_pattern_cannot_stand_for_two_planted_themes():
    golden = {"themes": [
        {"id": "a", "label": "A", "participants": ["P1", "P2", "P3"], "keywords": ["nota"]},
        {"id": "b", "label": "B", "participants": ["P1", "P2", "P3", "P4"], "keywords": ["nota"]},
    ]}
    ev = evals.score(result_with([theme("Nota fiscal", ["P1", "P2", "P3"])], []), golden)
    assert sum(1 for m in ev["theme_matches"] if m["matched_theme"]) == 1
    assert ev["theme_recall"] == 0.5


def test_score_flags_pii_leak():
    ev = evals.score(result_with([], [insight("P2", claim="x@exemplo.com")]), GOLDEN)
    assert ev["pii_leaks"] == ["email"] and ev["theme_recall"] == 0.0


def test_isolated_themes_never_count_as_matches():
    th = [theme("Atraso", ["P1", "P2"], status="isolated")]
    assert evals.score(result_with(th, []), GOLDEN)["theme_recall"] == 0.0


@pytest.mark.parametrize("raw, expected", [
    ("Clientes atrasam. Cobrar é difícil. (P1-1, P1-2, P2-2, P4-2,P", "Clientes atrasam. Cobrar é difícil."),
    ("Clientes atrasam e cobrar é difícil, especialmente em", "…"),
    ("Tudo certo aqui.", "Tudo certo aqui."),
    ("P1 e P2 relatam atraso dos clientes.", "P1 e P2 relatam atraso dos clientes."),
])
def test_clean_summary_handles_cut_text(raw, expected):
    out = themes.clean_summary(raw)
    assert out.endswith(expected) and "P1-" not in out and len(out) >= len(expected)


def test_summary_id_references_are_stripped():
    raw = [{"name": "A", "summary": "Pessoas atrasam (P1-1, P9-9).", "insight_ids": ["P1-1"]}]
    assert themes.validate(raw, make_insights())[0]["summary"] == "Pessoas atrasam."


# ---------- LLM client (fake server) ----------

class FakeServer:
    def __init__(self, content, finish_reason="stop"):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                payload = {"model": body["model"], "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
MSGS = [{"role": "user", "content": "oi"}]


def test_llm_sends_schema_logs_and_caches(tmp_path):
    srv = FakeServer('{"a": 1}')
    try:
        llm = LocalLLM("m", tmp_path, base_url=srv.url)
        assert llm.json("t", MSGS, SCHEMA) == {"a": 1}
        assert llm.json("t", MSGS, SCHEMA) == {"a": 1}       # second call served from cache
        assert len(srv.requests) == 1
        sent = srv.requests[0]
        assert sent["response_format"]["type"] == "json_schema"
        assert sent["response_format"]["json_schema"]["schema"] == SCHEMA
        assert sent["temperature"] == 0.0
        log = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        assert log[0]["completion_tokens"] == 5
    finally:
        srv.close()


def test_llm_offline_replays_cache_and_fails_without_it(tmp_path):
    srv = FakeServer('{"a": 2}')
    try:
        LocalLLM("m", tmp_path, base_url=srv.url).json("t", MSGS, SCHEMA)
    finally:
        srv.close()
    offline = LocalLLM("m", tmp_path, offline=True)
    assert offline.json("t", MSGS, SCHEMA) == {"a": 2}
    with pytest.raises(LLMError):
        offline.json("t", [{"role": "user", "content": "outra"}], SCHEMA)


def test_llm_invalid_json_raises(tmp_path):
    srv = FakeServer("isto não é json")
    try:
        with pytest.raises(LLMError):
            LocalLLM("m", tmp_path, base_url=srv.url).json("t", MSGS, SCHEMA)
    finally:
        srv.close()


def test_llm_truncated_output_is_an_error_and_not_cached(tmp_path):
    srv = FakeServer('{"a": [1, 1, 1, 1', finish_reason="length")
    try:
        llm = LocalLLM("m", tmp_path, base_url=srv.url)
        with pytest.raises(LLMError, match="token limit"):
            llm.json("t", MSGS, SCHEMA, max_tokens=50)
        assert srv.requests[0]["max_tokens"] == 50
        assert not list((tmp_path / "cache").iterdir())
    finally:
        srv.close()


def test_llm_unreachable_server_has_clear_error(tmp_path):
    with pytest.raises(LLMError, match="LM Studio"):
        LocalLLM("m", tmp_path, base_url="http://127.0.0.1:9/v1", timeout=2).json("t", MSGS, SCHEMA)


# ---------- report ----------

def test_report_escapes_model_text(tr):
    insight = {"id": "P9-1", "participant": "P9", "claim": "<script>alert(1)</script>", "kind": "dor",
               "quote": "já baixei dois desses e larguei os dois", "turn": 2, "status": "verified"}
    result = {
        "meta": {"model": "m", "created": "agora", "interviews": 1, "pattern_min": 3},
        "transcripts": [{"id": "P9", "profile": "", "turns": [
            {"n": t.n, "speaker": t.speaker, "is_participant": t.is_participant, "text": t.text} for t in tr.turns]}],
        "redactions": {}, "insights": [insight],
        "themes": [{"name": "T", "summary": "s", "insight_ids": ["P9-1"], "participants": ["P9"], "status": "isolated"}],
        "calls": [],
    }
    html = report.render(result)
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "Sinais isolados" in html
