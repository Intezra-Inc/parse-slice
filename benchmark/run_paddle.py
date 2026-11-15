"""Run PaddleOCR over the crops or the full pages; one JSON line per image with the text and the seconds taken.

ppocr / ppocr6: PP-OCRv5 server / PP-OCRv6 medium detection + recognition with text-line orientation on; document orientation and unwarping
       off (neither model gets page-level geometry correction). Text = recognised lines in PaddleOCR's output order.
vl:    PaddleOCR-VL-1.6 document parser: layout model (Paddle, this GPU) + the 0.9B VLM served by vLLM 0.31
       (serve_vl.sh, Paddle's recommended backend; the native Paddle path ran single-threaded on the CPU). Its Markdown is reduced to
       plain text (tables' HTML tags, image links and Markdown markup removed) before scoring.

Usage: PADDLE_PDX_CACHE_HOME=cache-paddle venv-paddle/bin/python -I run_paddle.py <crops|pages> <ppocr|ppocr6|vl> <out.jsonl> [limit]
Resumes: ids already in out.jsonl are skipped.
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
D = os.path.dirname(os.path.abspath(__file__))
MODE, ENGINE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
items = [json.loads(x) for x in open(f"{D}/{MODE}.jsonl")][:LIMIT]
done = {json.loads(x)["id"] for x in open(OUT)} if os.path.exists(OUT) else set()
sep = " " if MODE == "crops" else "\n"

if ENGINE in ("ppocr", "ppocr6"):
    from paddleocr import PaddleOCR
    v = "PP-OCRv5_server" if ENGINE == "ppocr" else "PP-OCRv6_medium"
    ocr = PaddleOCR(text_detection_model_name=f"{v}_det", text_recognition_model_name=f"{v}_rec",
                    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True,
                    device="gpu:0")

    def run(path):
        r = ocr.predict(path)[0]
        return sep.join(r["rec_texts"])
else:
    from paddleocr import PaddleOCRVL
    ocr = PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="vllm-server",
                      vl_rec_server_url=os.environ.get("VL_URL", "http://127.0.0.1:8118/v1"),
                      vl_rec_api_model_name="PaddleOCR-VL-1.6-0.9B",
                      use_doc_orientation_classify=False, use_doc_unwarping=False, markdown_ignore_labels=[],
                      device="gpu:0")  # markdown_ignore_labels=[]: keep headers/footers/page numbers (in the GT)

    def run(path):
        r = ocr.predict(path)[0]
        md = r.markdown["markdown_texts"] if isinstance(r.markdown, dict) else str(r.markdown)
        md = re.sub(r"<img[^>]*>|!\[[^\]]*\]\([^)]*\)", " ", md)
        md = re.sub(r"<[^>]+>", " ", md)
        md = re.sub(r"^#+\s*|\*\*|__|`", "", md, flags=re.M)
        return md

with open(OUT, "a") as out:
    for it in items:
        if it["id"] in done:
            continue
        path = f"{D}/crops/{it['id']}.png" if MODE == "crops" else f"{D}/dl/omnidocbench/images/{it['image']}"
        t = time.perf_counter()
        try:
            text, err = run(path), None
        except Exception as e:  # noqa: BLE001
            text, err = "", repr(e)[:300]
        out.write(json.dumps({"id": it["id"], "text": text, "secs": round(time.perf_counter() - t, 4),
                              "error": err}, ensure_ascii=False) + "\n")
print("done", OUT)
