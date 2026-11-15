"""Run Nemotron 3.5 Parse over the full pages through its vLLM server (serve_parse.sh) with NVIDIA's DocumentParser
client; one JSON line per page with the plain text and the seconds taken.

mode: NVIDIA's default single pass (one full-page request per page). Markdown is reduced to plain text exactly as
for PaddleOCR-VL (HTML tags, image links and Markdown markup removed) before scoring.

Usage: venv-vllm/bin/python -I run_parse.py <out.jsonl> [limit] [single|double]
Resumes: ids already in out.jsonl are skipped.
"""
import json
import os
import re
import sys
import time

from nemotron_parse.inference import DocumentParser

D = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
MODE = sys.argv[3] if len(sys.argv) > 3 else "single"
items = [json.loads(x) for x in open(f"{D}/pages.jsonl")][:LIMIT]
done = {json.loads(x)["id"] for x in open(OUT)} if os.path.exists(OUT) else set()


def plain(md):
    md = re.sub(r"<img[^>]*>|!\[[^\]]*\]\([^)]*\)", " ", md)
    md = re.sub(r"<[^>]+>", " ", md)
    return re.sub(r"^#+\s*|\*\*|__|`", "", md, flags=re.M)


with DocumentParser(base_url=os.environ.get("PARSE_URL", "http://127.0.0.1:8119/v1"), model=os.environ.get("PARSE_MODEL", "nemotron-parse")) as parser, \
        open(OUT, "a") as out:
    for it in items:
        if it["id"] in done:
            continue
        t = time.perf_counter()
        try:
            text, err = plain(parser.parse(f"{D}/dl/omnidocbench/images/{it['image']}", mode=MODE)["markdown"]), None
        except Exception as e:  # noqa: BLE001
            text, err = "", repr(e)[:300]
        out.write(json.dumps({"id": it["id"], "text": text, "secs": round(time.perf_counter() - t, 4),
                              "error": err}, ensure_ascii=False) + "\n")
print("done", OUT)
