"""Run a general-purpose vision-language model as an OCR engine over the full pages, through an OpenAI-compatible
vLLM server; one JSON line per page with the plain text, the seconds taken and the token counts.

Same prompt for every model (below), image first then text (Gemma 4's recommended order), greedy decoding
(temperature 0), thinking off, up to 16,384 output tokens. The Markdown answer is reduced to plain text exactly as for
PaddleOCR-VL and Nemotron 3.5 Parse before scoring. Pages run CONC at a time, so `secs` is the time per request under
that load, not single-page latency; throughput is pages / wall-clock (printed at the end and in the .meta.json file).

Sampling overrides (default greedy): TEMP, TOP_P, TOP_K env vars, e.g. a model card's recommended sampling.
IDS=<file of page ids, one per line> runs only those pages.

Usage: venv-vllm/bin/python -I run_vlm.py <url> <served model> <out.jsonl> [concurrency=32] [limit]
Resumes: ids already in out.jsonl are skipped.
"""
import base64
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from PIL import Image

PROMPT = ("Transcribe all the text on this page in natural reading order. Output Markdown: use headings, paragraphs "
          "and lists as they appear on the page, tables as HTML, and formulas as LaTeX. Include headers, footers, page "
          "numbers, captions and footnotes. Do not describe images, do not summarize, and do not add anything that "
          "is not on the page.")
D = os.path.dirname(os.path.abspath(__file__))
URL, MODEL, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 32
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))
items = [json.loads(x) for x in open(f"{D}/pages.jsonl")][:LIMIT]
done = {json.loads(x)["id"] for x in open(OUT)} if os.path.exists(OUT) else set()
todo = [it for it in items if it["id"] not in done]
if os.environ.get("IDS"):
    keep = {x.strip() for x in open(os.environ["IDS"]) if x.strip()}
    todo = [it for it in todo if it["id"] in keep]
SAMPLING = {"temperature": float(os.environ.get("TEMP", "0"))}
if os.environ.get("TOP_P"):
    SAMPLING["top_p"] = float(os.environ["TOP_P"])
if os.environ.get("TOP_K"):
    SAMPLING["top_k"] = int(os.environ["TOP_K"])


def plain(md):  # same Markdown -> text reduction as run_parse.py / run_paddle.py (vl)
    md = re.sub(r"<img[^>]*>|!\[[^\]]*\]\([^)]*\)", " ", md)
    md = re.sub(r"<[^>]+>", " ", md)
    return re.sub(r"^#+\s*|\*\*|__|`", "", md, flags=re.M)


def data_url(path):
    with Image.open(path) as im:
        buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


client = httpx.Client(timeout=1800, limits=httpx.Limits(max_connections=CONC * 2))
lock = threading.Lock()
out = open(OUT, "a")


def one(it):
    body = {"model": MODEL, **SAMPLING, "max_tokens": MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(f"{D}/dl/omnidocbench/images/{it['image']}")}},
                {"type": "text", "text": PROMPT}]}]}
    t = time.perf_counter(); text, err, usage, fin = "", None, {}, None
    try:
        r = client.post(f"{URL}/chat/completions", json=body); r.raise_for_status(); j = r.json()
        ch = j["choices"][0]; fin = ch.get("finish_reason"); usage = j.get("usage") or {}
        text = plain(ch["message"].get("content") or "")
        if fin == "length":
            err = "finish_reason=length (output kept)"
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:300]
    row = {"id": it["id"], "text": text, "secs": round(time.perf_counter() - t, 3), "error": err, "finish": fin,
           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
    with lock:
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()


t0 = time.perf_counter()
with ThreadPoolExecutor(CONC) as ex:
    list(ex.map(one, todo))
wall = time.perf_counter() - t0
meta = {"model": MODEL, "url": URL, "concurrency": CONC, "pages": len(todo), "wall_s": round(wall, 1),
        "pages_per_s": round(len(todo) / wall, 3) if todo else None, "max_tokens": MAX_TOKENS, "sampling": SAMPLING, "prompt": PROMPT}
json.dump(meta, open(OUT.replace(".jsonl", ".meta.json"), "w"), indent=1)
print(json.dumps(meta))
