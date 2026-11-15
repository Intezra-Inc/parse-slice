"""Score the full two-machine OCR benchmark (run_full.sh) and the general-VLM runs (run_vlm.py).

Accuracy is scored exactly as in score.py (token recall / precision, page NED on 1,645 pages; crop NED with
OmniDocBench's clean_string, empties skipped). Speed per machine: pages/s = pages / sum(seconds), sequential, first
5 pages dropped; reported for the first 300 pages and for all pages. VLM runs were concurrent: their pages/s is
wall-clock throughput from the .meta.json file.

Usage: venv-nemo/bin/python -I score_full.py > SCORES_FULL.md   (also writes scores_full.json)
"""
import collections
import json
import os
import re
import statistics as st
import unicodedata

from rapidfuzz.distance import Levenshtein

D = os.path.dirname(os.path.abspath(__file__))
P = f"{D}/preds_full"
BOXES = [("h100", "H100 NVL"), ("blackwell", "RTX PRO 6000 Blackwell Max-Q")]
ENG = [("nemo-multi", "Nemotron OCR v2, multilingual"), ("nemo-en", "Nemotron OCR v2, English"),
       ("ppocr", "PP-OCRv5 server"), ("ppocr6", "PP-OCRv6 medium"), ("paddlevl", "PaddleOCR-VL-1.6"),
       ("parse35", "Nemotron 3.5 Parse"), ("parse35d", "Nemotron 3.5 Parse, double pass"),
       ("parse35s", "Nemotron 3.5 Parse + slicing (parse-slice)")]
VLM = [("n35vl", "Nemotron 3.5 Super VL (2 GPUs, NVFP4)", 2), ("gemma31", "Gemma 4 31B (BF16)", 1),
       ("gemma26", "Gemma 4 26B-A4B (BF16)", 1)]
TOK = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[^\W_]+")


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def norm_odb(s):
    return re.sub(r"[^\w一-鿿]", "", s or "")


def ned(p, g, f=norm):
    p, g = f(p), f(g)
    return 0.0 if not p and not g else Levenshtein.distance(p, g) / max(len(p), len(g))


toks = lambda s: collections.Counter(TOK.findall(unicodedata.normalize("NFKC", s or "").lower()))
mean = lambda xs: round(st.mean(xs), 3) if xs else None
gt_p = {r["id"]: r for r in jl(f"{D}/pages.jsonl")}
gt_c = {r["id"]: r for r in jl(f"{D}/crops.jsonl")}


def pages(rows):
    per = []
    for r in rows:
        g = gt_p.get(r["id"])
        if not g or not norm(g["text"]):
            continue
        gt, pt = toks(g["text"]), toks(r["text"])
        hit = sum((gt & pt).values())
        rec = hit / max(1, sum(gt.values())); prec = hit / max(1, sum(pt.values())) if pt else 0.0
        e = r.get("error") or ""
        per.append((g, rec, prec, ned(r["text"], g["text"]), bool(e) and "output kept" not in e))
    return {"n": len(per), "recall": mean([x[1] for x in per]), "precision": mean([x[2] for x in per]),
            "ned": mean([x[3] for x in per]), "recall_en": mean([x[1] for x in per if x[0]["language"] == "english"]),
            "recall_zh": mean([x[1] for x in per if x[0]["language"] == "simplified_chinese"]),
            "failed": sum(x[4] for x in per)}


def crops(rows):
    sc = [ned(r["text"], gt_c[r["id"]]["text"], norm_odb) for r in rows if r["id"] in gt_c and norm(r["text"])]
    en = [ned(r["text"], gt_c[r["id"]]["text"], norm_odb) for r in rows
          if r["id"] in gt_c and norm(r["text"]) and gt_c[r["id"]]["language"] == "text_english"]
    return {"n": len(rows), "ned": mean(sc), "ned_en": mean(en)}


def speed(rows, n=None):
    r = rows[5:n]
    return round(len(r) / sum(x["secs"] for x in r), 2) if r else None


out = {}
L = ["# OCR benchmark, full run on two machines (OmniDocBench)", ""]
L += ["## Full pages: accuracy (1,645 pages)", "",
      "| engine | machine | words found | precision | EN | ZH | reading-order error | failed pages |", "|---" * 8 + "|"]
for k, name in ENG:
    for b, bname in BOXES:
        rows = jl(f"{P}/{b}/{k}_pages.jsonl")
        if not rows:
            continue
        s = pages(rows); out[f"{b}/{k}_pages"] = s
        L.append(f"| {name} | {bname} | **{s['recall']}** | {s['precision']} | {s['recall_en']} | {s['recall_zh']} | "
                 f"{s['ned']} | {s['failed']} |")
for k, name, g in VLM:
    rows = jl(f"{P}/vlm/{k}_pages.jsonl")
    if rows:
        s = pages(rows); out[f"vlm/{k}_pages"] = s
        L.append(f"| {name} | (VLM run, greedy) | **{s['recall']}** | {s['precision']} | {s['recall_en']} | {s['recall_zh']} | "
                 f"{s['ned']} | {s['failed']} |")
        # pages that ran into the 16,384-token cap under greedy decoding, rerun with the model card's sampling
        rs = {r["id"]: r for r in jl(f"{P}/vlm/{k}_resample.jsonl")}
        if rs:
            merged = [rs.get(r["id"], r) for r in rows]
            s = pages(merged); out[f"vlm/{k}_pages_resampled"] = s
            s["still_capped"] = sum(r.get("finish") == "length" for r in rs.values())
            L.append(f"| {name}, looping pages resampled ({len(rs)}; {s['still_capped']} still capped) | (VLM run) | "
                     f"**{s['recall']}** | {s['precision']} | {s['recall_en']} | {s['recall_zh']} | {s['ned']} | {s['failed']} |")
L += ["", "## Text blocks (23,078, OmniDocBench v1.5), edit distance (lower is better)", "",
      "| engine | machine | all | English |", "|---|---|---|---|"]
for k, name in ENG[:4]:
    for b, bname in BOXES:
        rows = jl(f"{P}/{b}/{k}_crops.jsonl")
        if rows:
            s = crops(rows); out[f"{b}/{k}_crops"] = s
            L.append(f"| {name} | {bname} | {s['ned']} | {s['ned_en']} |")
L += ["", "## Speed, one page at a time, one GPU (pages/s; first 5 pages dropped)", "",
      "| engine | H100, first 300 | Blackwell, first 300 | H100, all pages | Blackwell, all pages | Blackwell / H100 (all) |",
      "|---|---|---|---|---|---|"]
for k, name in ENG:
    h, bl = jl(f"{P}/h100/{k}_pages.jsonl"), jl(f"{P}/blackwell/{k}_pages.jsonl")
    v = [speed(h, 300), speed(bl, 300), speed(h), speed(bl)]
    out[f"speed/{k}"] = v
    ratio = round(v[3] / v[2], 2) if v[2] and v[3] else None
    L.append(f"| {name} | {v[0]} | {v[1]} | {v[2]} | {v[3]} | {ratio} |")
L += ["", "## General VLMs: throughput (concurrent, 32 requests in flight)", "",
      "| model | GPUs | pages/s | pages/s per GPU | pages hitting the 16,384-token cap |", "|---|---|---|---|---|"]
for k, name, g in VLM:
    m = f"{P}/vlm/{k}_pages.meta.json"
    rows = jl(f"{P}/vlm/{k}_pages.jsonl")
    if os.path.exists(m) and rows:
        t = json.load(open(m))["pages_per_s"]
        L.append(f"| {name} | {g} | {t} | {round(t / g, 3)} | {sum(r.get('finish') == 'length' for r in rows)} |")
L += ["", "## Reproducibility: H100 vs Blackwell (same software, same pages)", "",
      "| engine | pages with identical text | words found, H100 - Blackwell |", "|---|---|---|"]
for k, name in ENG:
    h = {r["id"]: r["text"] for r in jl(f"{P}/h100/{k}_pages.jsonl")}
    bl = {r["id"]: r["text"] for r in jl(f"{P}/blackwell/{k}_pages.jsonl")}
    ids = set(h) & set(bl)
    if ids and f"h100/{k}_pages" in out and f"blackwell/{k}_pages" in out:
        same = sum(h[i] == bl[i] for i in ids)
        d = round(out[f"h100/{k}_pages"]["recall"] - out[f"blackwell/{k}_pages"]["recall"], 3)
        L.append(f"| {name} | {same}/{len(ids)} | {d:+} |")
json.dump(out, open(f"{D}/scores_full.json", "w"), indent=1)
print("\n".join(L))
