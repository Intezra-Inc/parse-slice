"""Score the OCR benchmark; writes SCORES.md (stdout) and scores.json.

Text normalisation for every metric: Unicode NFKC (full-width punctuation -> ASCII), lowercase for tokens.
crops: normalised edit distance (NED, lower is better) per block after OmniDocBench's own clean_string (keeps only
       word characters and CJK, i.e. drops whitespace and punctuation; a punctuation-kept column is also shown),
       averaged over blocks
       ("sample_avg"), split by OmniDocBench's text_language / text_background / text_rotate attributes like NVIDIA's
       model-card table. Main number skips empty predictions as NVIDIA's table does; the "empties = 1.0" column
       counts them as complete misses.
pages: per page, against the ground-truth text in reading order:
       NED (whitespace removed; order-sensitive, so it also penalises reading order)
       token recall / precision / F1 (multiset of words, CJK counted per character; order-insensitive).
       GT excludes table bodies, formulas and figure text, so precision is understated for every engine (they all
       also read table cells); recall is the cleaner accuracy number.
speed: preds/speed_<engine>.jsonl, the first 300 pages run one engine at a time on one H100 NVL with nothing else on
       the box; first 5 pages dropped as warm-up.

Usage: venv-nemo/bin/python -I score.py
"""
import collections
import json
import os
import re
import statistics as st
import unicodedata

from rapidfuzz.distance import Levenshtein

D = os.path.dirname(os.path.abspath(__file__))
ENGINES = [("nemo-multi", "Nemotron OCR v2 (multilingual)"), ("nemo-en", "Nemotron OCR v2 (English)"),
           ("ppocr", "PP-OCRv5 server"), ("ppocr6", "PP-OCRv6 medium"), ("paddlevl", "PaddleOCR-VL-1.6"), ("parse35", "Nemotron 3.5 Parse"), ("parse35d", "Nemotron 3.5 Parse (double pass)")]
TOK = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[^\W_]+")


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def norm_odb(s):
    """OmniDocBench's clean_string: drop everything except word characters and CJK (removes punctuation too)."""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s or "")


def ned(p, g, f=norm):
    p, g = f(p), f(g)
    if not p and not g:
        return 0.0
    return Levenshtein.distance(p, g) / max(len(p), len(g))


def toks(s):
    return collections.Counter(TOK.findall(unicodedata.normalize("NFKC", s or "").lower()))


def mean(xs):
    return round(st.mean(xs), 3) if xs else None


gt_c = {r["id"]: r for r in jl(f"{D}/crops.jsonl")}
gt_p = {r["id"]: r for r in jl(f"{D}/pages.jsonl")}
out = {}
CROP_SPLITS = [("language", "text_english", "EN"), ("language", "text_simplified_chinese", "ZH"),
               ("language", "text_en_ch_mixed", "Mixed"), ("background", "white", "White"),
               ("background", "single_colored", "Single"), ("background", "multi_colored", "Multi"),
               ("rotate", "normal", "Normal"), ("rotate", "rotate90", "Rotate90"), ("rotate", "rotate270", "Rotate270"),
               ("rotate", "horizontal", "Horizontal")]

lines = ["# OCR benchmark: Nemotron OCR v2 vs PaddleOCR (OmniDocBench)", ""]
lines += ["## Crop mode (23,078 text blocks, OmniDocBench v1.5 attributes), NED sample_avg, lower is better", "",
          "| engine | blocks | all | all, punctuation kept | all (empties = 1.0) | empty | " + " | ".join(s[2] for s in CROP_SPLITS) + " | s/block |",
          "|---" * (7 + len(CROP_SPLITS)) + "|"]
for key, name in ENGINES:
    pr = jl(f"{D}/preds/{key}_crops.jsonl")
    if not pr:
        continue
    rows = [(gt_c[r["id"]], r) for r in pr if r["id"] in gt_c]
    scored = [(g, ned(r["text"], g["text"], norm_odb)) for g, r in rows if norm(r["text"])]
    strict = [ned(r["text"], g["text"]) for g, r in rows if norm(r["text"])]
    empty = len(rows) - len(scored)
    res = {"n": len(rows), "empty": empty, "ned": mean([x for _, x in scored]), "ned_strict": mean(strict),
           "ned_with_empty": mean([x for _, x in scored] + [1.0] * empty),
           "secs": mean([r["secs"] for _, r in rows])}
    for f, v, lab in CROP_SPLITS:
        res[lab] = mean([x for g, x in scored if g[f] == v])
    out[f"{key}_crops"] = res
    lines.append(f"| {name} | {res['n']} | **{res['ned']}** | {res['ned_strict']} | {res['ned_with_empty']} | {empty} | "
                 + " | ".join(str(res[s[2]]) for s in CROP_SPLITS) + f" | {res['secs']} |")
lines.append("")

lines += ["## Full pages (1,651 pages, OmniDocBench v1.6)", "",
          "| engine | pages | token recall | token precision | token F1 | page NED (order-sensitive) | recall EN | recall ZH | "
          "recall single-column | recall multi-column | errors |", "|---" * 11 + "|"]
for key, name in ENGINES:
    pr = jl(f"{D}/preds/{key}_pages.jsonl")
    if not pr:
        continue
    per = []
    for r in pr:
        g = gt_p.get(r["id"])
        if not g or not norm(g["text"]):
            continue
        gt, pt = toks(g["text"]), toks(r["text"])
        hit = sum((gt & pt).values())
        rec = hit / max(1, sum(gt.values()))
        prec = hit / max(1, sum(pt.values())) if pt else 0.0
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
        per.append((g, rec, prec, f1, ned(r["text"], g["text"]), bool(r.get("error"))))
    multi = {"double_column", "three_column", "1andmore_column"}
    res = {"n": len(per), "recall": mean([x[1] for x in per]), "precision": mean([x[2] for x in per]),
           "f1": mean([x[3] for x in per]), "ned": mean([x[4] for x in per]),
           "recall_en": mean([x[1] for x in per if x[0]["language"] == "english"]),
           "recall_zh": mean([x[1] for x in per if x[0]["language"] == "simplified_chinese"]),
           "recall_single": mean([x[1] for x in per if x[0]["layout"] == "single_column"]),
           "recall_multi": mean([x[1] for x in per if x[0]["layout"] in multi]),
           "errors": sum(x[5] for x in per)}
    out[f"{key}_pages"] = res
    lines.append(f"| {name} | {res['n']} | **{res['recall']}** | {res['precision']} | {res['f1']} | {res['ned']} | "
                 f"{res['recall_en']} | {res['recall_zh']} | {res['recall_single']} | {res['recall_multi']} | "
                 f"{res['errors']} |")
lines.append("")

lines += ["## Speed (first 300 pages, one engine at a time, one H100 NVL)", "",
          "| engine | pages/s | s/page p50 | s/page p95 |", "|---|---|---|---|"]
for key, name in ENGINES:
    pr = jl(f"{D}/preds/speed_{key}.jsonl")[5:]
    if not pr:
        continue
    s = sorted(r["secs"] for r in pr)
    res = {"pages_per_s": round(len(s) / sum(s), 2), "p50": round(s[len(s) // 2], 3),
           "p95": round(s[int(len(s) * 0.95)], 3)}
    out[f"{key}_speed"] = res
    lines.append(f"| {name} | **{res['pages_per_s']}** | {res['p50']} | {res['p95']} |")

json.dump(out, open(f"{D}/scores.json", "w"), indent=1)
print("\n".join(lines))
