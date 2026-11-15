"""Run Nemotron 3.5 Parse + page slicing (the parse-slice package in release/parse-slice) over the full pages,
one page at a time, in the same output format as run_parse.py plus how each page was cut.

Usage: venv-vllm/bin/python -I run_slice.py <out.jsonl> [limit or ""]
Server: PARSE_URL (default http://127.0.0.1:8119/v1). Resumes: ids already in out.jsonl are skipped.
"""
import json
import os
import re
import sys

from parse_slice import SlicingParser

D = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
items = [json.loads(x) for x in open(f"{D}/pages.jsonl")][:LIMIT]
done = {json.loads(x)["id"] for x in open(OUT)} if os.path.exists(OUT) else set()


def plain(md):  # same Markdown -> text reduction as run_parse.py
    md = re.sub(r"<img[^>]*>|!\[[^\]]*\]\([^)]*\)", " ", md)
    md = re.sub(r"<[^>]+>", " ", md)
    return re.sub(r"^#+\s*|\*\*|__|`", "", md, flags=re.M)


with SlicingParser(os.environ.get("PARSE_URL", "http://127.0.0.1:8119/v1"), "nemotron-parse") as sp, open(OUT, "a") as out:
    for it in items:
        if it["id"] in done:
            continue
        r = sp.parse(f"{D}/dl/omnidocbench/images/{it['image']}")
        err = None if r.complete else "; ".join(p.error for p in r.pieces if not p.ok)[:300]
        out.write(json.dumps({"id": it["id"], "text": plain(r.markdown), "secs": r.seconds, "error": err,
                              "pieces": len(r.pieces), "requests": r.requests}, ensure_ascii=False) + "\n")
print("done", OUT)
