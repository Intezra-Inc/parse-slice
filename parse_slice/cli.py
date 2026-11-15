"""parse-slice: turn images and PDFs into Markdown with Nemotron 3.5 Parse, slicing pages that run out of room.

    parse-slice INPUT [INPUT ...] -o OUTPUT_DIR [--url URL] [--concurrency N] [--mode single|double]

INPUT is a file or a directory (searched recursively) of .png .jpg .jpeg .tif .tiff .webp .bmp and .pdf files.
For each image, OUTPUT_DIR/<name>.md is written; for each PDF, OUTPUT_DIR/<name>.md with all pages in order, separated
by a page-break comment. OUTPUT_DIR/results.jsonl gets one line per page: how it was cut, requests, seconds, errors.
Files whose output already exists are skipped, so an interrupted run can be resumed by running it again.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .core import SlicingParser

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def find_inputs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT | {".pdf"})
        elif p.suffix.lower() in IMAGE_EXT | {".pdf"}:
            out.append(p)
        else:
            sys.exit(f"parse-slice: not an image, PDF or directory: {p}")
    return out


def pdf_pages(path: Path, dpi: int):
    try:
        import pypdfium2 as pdfium
    except ImportError:
        sys.exit("parse-slice: PDF input needs pypdfium2 (pip install 'parse-slice[pdf]')")
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(pdf)):
            yield i + 1, pdf[i].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        pdf.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="parse-slice", description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="image or PDF files, or directories of them")
    ap.add_argument("-o", "--output", required=True, help="directory for the Markdown files and results.jsonl")
    ap.add_argument("--url", default="http://localhost:8000/v1", help="Nemotron 3.5 Parse server (default %(default)s)")
    ap.add_argument("--model", default="nemotron-parse", help="served model name (default %(default)s)")
    ap.add_argument("--mode", default="single", choices=["single", "double"],
                    help="single: one request per page (default). double: also re-reads tables and charts")
    ap.add_argument("--concurrency", type=int, default=64, help="pages in flight (default %(default)s)")
    ap.add_argument("--max-depth", type=int, default=3, help="max cuts per page piece (default %(default)s: 8 pieces)")
    ap.add_argument("--no-slice", action="store_true", help="never cut pages (NVIDIA's behaviour: failed pages stay empty)")
    ap.add_argument("--pdf-dpi", type=int, default=200, help="render resolution for PDF pages (default %(default)s)")
    ap.add_argument("--max-megapixels", type=float, default=400,
                    help="largest input image accepted, in megapixels (default %(default)s; Pillow's own limit is ~179)")
    a = ap.parse_args(argv)
    Image.MAX_IMAGE_PIXELS = int(a.max_megapixels * 1e6)  # large scans are expected input here, not an attack

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    files = [f for f in find_inputs(a.inputs) if not (out / f"{f.stem}.md").exists()]
    if not files:
        print("parse-slice: nothing to do (all outputs exist)", file=sys.stderr)
        return 0
    lock = threading.Lock()
    log = open(out / "results.jsonl", "a")
    stats = {"pages": 0, "sliced": 0, "incomplete": 0}
    t0 = time.perf_counter()

    with SlicingParser(a.url, a.model, mode=a.mode, max_depth=0 if a.no_slice else a.max_depth) as sp:
        def do_page(src: Path, page_no, img):
            r = sp.parse(img)
            row = {"file": str(src), "page": page_no, **r.to_dict()}
            with lock:
                log.write(json.dumps(row, default=list) + "\n"); log.flush()
                stats["pages"] += 1; stats["sliced"] += r.sliced; stats["incomplete"] += not r.complete
                print(f"\r{stats['pages']} pages, {stats['sliced']} sliced, {stats['incomplete']} incomplete, "
                      f"{stats['pages'] / (time.perf_counter() - t0):.2f} pages/s", end="", file=sys.stderr, flush=True)
            return r.markdown

        with ThreadPoolExecutor(a.concurrency) as pool:
            # submit every page first so up to --concurrency pages are in flight, then write files as they finish
            jobs = {}
            for f in files:
                if f.suffix.lower() == ".pdf":
                    jobs[f] = [pool.submit(do_page, f, n, img) for n, img in pdf_pages(f, a.pdf_dpi)]
                else:
                    jobs[f] = [pool.submit(do_page, f, None, f)]
            owner = {fu: f for f, futs in jobs.items() for fu in futs}
            left = {f: len(futs) for f, futs in jobs.items()}
            for fu in as_completed(owner):
                f = owner[fu]
                left[f] -= 1
                if left[f]:
                    continue
                pages = [x.result() for x in jobs[f]]
                text = (pages[0] + "\n" if f.suffix.lower() != ".pdf" else
                        "".join(f"<!-- page {i} -->\n\n{md}\n\n" for i, md in enumerate(pages, 1)))
                (out / f"{f.stem}.md").write_text(text)
    log.close()
    print(f"\nparse-slice: {stats['pages']} pages in {time.perf_counter() - t0:.1f} s; {stats['sliced']} sliced; "
          f"{stats['incomplete']} with a piece that still failed (see results.jsonl)", file=sys.stderr)
    return 1 if stats["incomplete"] else 0


if __name__ == "__main__":
    sys.exit(main())
