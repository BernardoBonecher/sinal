"""Score a run against ground truth planted in the synthetic interviews (evals/golden.json).

Metrics:
- faithfulness      share of the model's quotes that are literal participant speech (before filtering)
- status breakdown  verified / paraphrased / not_found / interviewer_words / too_short
- theme recall      planted themes that some reported pattern recovers
- theme precision   reported patterns that match a planted theme (an unplanted but real theme
                    also counts against it, so read the unmatched ones)
- grouped_share     verified insights the model placed in some theme
- counter-evidence  participants placed in a theme they explicitly contradict
- pii_leaks         planted personal data that appears anywhere in the run output
- speed             model time and tokens per second (local, CPU or GPU)
"""

import json

MATCH_MIN = 0.6   # overlap coefficient between participant sets
MATCH_MIN_SHARED = 2
TOPIC_MIN = 0.5   # share of the theme's insights that must be about the planted subject


def overlap(a: set, b: set) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def topic_share(theme: dict, insights: dict, keywords: list[str]) -> float:
    """Share of the theme's insights whose claim or quote mentions one of the keywords."""
    texts = [(insights[i]["claim"] + " " + insights[i]["quote"]).lower()
             for i in theme["insight_ids"] if i in insights]
    if not texts:  # no insight text available: judge by the theme's own name and summary
        texts = [(theme["name"] + " " + theme.get("summary", "")).lower()]
    return sum(any(k in t for k in keywords) for t in texts) / len(texts)


def match_themes(themes: list[dict], golden: dict, insights: dict) -> list[dict]:
    """One-to-one matching of planted themes to reported patterns.

    A pair is a candidate when the participant sets overlap enough AND at least TOPIC_MIN of the
    reported theme's insights are about the planted subject (mention one of its keywords). Pairs
    are then taken greedily by Jaccard, so one pattern can't stand for two planted themes.
    """
    patterns = [t for t in themes if t["status"] == "pattern"]
    candidates = []
    for gi, g in enumerate(golden["themes"]):
        gp = set(g["participants"])
        for ti, t in enumerate(patterns):
            tp = set(t["participants"])
            shared = gp & tp
            s = overlap(gp, tp)
            jaccard = len(shared) / len(gp | tp)
            topic = topic_share(t, insights, g.get("keywords", [""]))
            if s >= MATCH_MIN and len(shared) >= MATCH_MIN_SHARED and topic >= TOPIC_MIN:
                # Rank by Jaccard so the theme covering more of the right people wins over a
                # smaller subset that the overlap coefficient would score as a tie.
                candidates.append((jaccard, topic, s, gi, ti))
    taken_g, taken_t, pairs = set(), set(), {}
    for _, _, s, gi, ti in sorted(candidates, reverse=True):
        if gi not in taken_g and ti not in taken_t:
            taken_g.add(gi)
            taken_t.add(ti)
            pairs[gi] = (patterns[ti], s)

    rows = []
    for gi, g in enumerate(golden["themes"]):
        best, s = pairs.get(gi, (None, 0.0))
        found = best["participants"] if best else []
        rows.append({
            "golden": g["id"], "label": g["label"], "expected": g["participants"],
            "matched_theme": best["name"] if best else None,
            "found": found,
            "score": round(s, 2),
            "people_recall": round(len(set(found) & set(g["participants"])) / len(g["participants"]), 2),
            "contradictions": sorted(set(found) & set(g.get("counter", []))),
        })
    return rows


def score(result: dict, golden: dict) -> dict:
    insights = result["insights"]
    statuses: dict[str, int] = {}
    for i in insights:
        statuses[i["status"]] = statuses.get(i["status"], 0) + 1
    kept = statuses.get("verified", 0) + statuses.get("verified_other_turn", 0)
    grouped = sum(len(t["insight_ids"]) for t in result["themes"] if t["status"] != "ungrouped")

    by_id = {i["id"]: i for i in insights}
    matches = match_themes(result["themes"], golden, by_id)
    patterns = [t for t in result["themes"] if t["status"] == "pattern"]
    matched_names = {m["matched_theme"] for m in matches if m["matched_theme"]}

    output_text = json.dumps(result, ensure_ascii=False)
    leaks = [p["kind"] for p in golden.get("pii", []) if p["value"] in output_text]

    # Leading-question traps. A hit is either a quote lifted from the interviewer (certain) or a
    # claim that matches the suggestion the participant rejected (heuristic, flagged for review).
    trap_hits = []
    for trap in golden.get("traps", []):
        rule = trap.get("suspect_if", {})
        for i in insights:
            if i["participant"] != trap["participant"]:
                continue
            claim = i["claim"].lower()
            if i["status"] == "interviewer_words":
                why = "quote taken from the interviewer"
            elif i["kind"] in rule.get("kind", []) and any(k in claim for k in rule.get("claim_contains", [])):
                why = "claim echoes the rejected suggestion (heuristic)"
            else:
                continue
            trap_hits.append({"trap": trap["id"], "insight": i["id"], "claim": i["claim"],
                              "status": i["status"], "why": why})

    calls = result.get("calls", [])
    seconds = sum(c.get("seconds") or 0 for c in calls)
    out_tokens = sum(c.get("completion_tokens") or 0 for c in calls)

    return {
        "insights": len(insights),
        "faithfulness": round(kept / len(insights), 3) if insights else None,
        "statuses": statuses,
        "theme_recall": round(sum(1 for m in matches if m["matched_theme"]) / len(matches), 3),
        "theme_precision": round(sum(1 for t in patterns if t["name"] in matched_names) / len(patterns), 3) if patterns else None,
        "patterns_reported": len(patterns),
        "patterns": [t["name"] for t in patterns],
        "grouped_share": round(grouped / kept, 3) if kept else None,
        "contradictions": sum(len(m["contradictions"]) for m in matches),
        "theme_matches": matches,
        "trap_hits": trap_hits,
        "pii_leaks": leaks,
        "model_seconds": round(seconds, 1),
        "output_tokens": out_tokens,
        "tokens_per_second": round(out_tokens / seconds, 1) if seconds else None,
    }
