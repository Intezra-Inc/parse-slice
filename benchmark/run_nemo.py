"""Run Nemotron OCR v2 over the crops or the full pages; one JSON line per image with the text and the seconds taken.

Text = the model's regions in the order it returns them (its relational model sets reading order), merged at
paragraph level; joined with a space for crops and a newline for pages.

Usage: venv-nemo/bin/python -I run_nemo.py <crops|pages> <multi|en> <out.jsonl> [limit]
Resumes: ids already in out.jsonl are skipped.
"""
import json
import os
import sys
import time

from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

D = os.path.dirname(os.path.abspath(__file__))
MODE, LANG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
items = [json.loads(x) for x in open(f"{D}/{MODE}.jsonl")][:LIMIT]
done = {json.loads(x)["id"] for x in open(OUT)} if os.path.exists(OUT) else set()
ocr = NemotronOCRV2(model_dir=f"{D}/dl/nemotron-ocr-v2/v2_{'multilingual' if LANG == 'multi' else 'english'}")
sep = " " if MODE == "crops" else "\n"
with open(OUT, "a") as out:
    for it in items:
        if it["id"] in done:
            continue
        path = f"{D}/crops/{it['id']}.png" if MODE == "crops" else f"{D}/dl/omnidocbench/images/{it['image']}"
        t = time.perf_counter()
        try:
            preds = ocr(path, merge_level="paragraph")
            text, err = sep.join(str(p["text"]) for p in preds if p.get("text")), None
        except Exception as e:  # noqa: BLE001
            text, err = "", repr(e)[:300]
        out.write(json.dumps({"id": it["id"], "text": text, "secs": round(time.perf_counter() - t, 4),
                              "error": err}, ensure_ascii=False) + "\n")
print("done", OUT)
