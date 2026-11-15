"""Build the OCR benchmark inputs from OmniDocBench.

crops: every text block in the v1.5 annotations that carries OCR attributes (text_language / text_background /
       text_rotate) and non-empty text -> crops/<id>.png (axis-aligned box of the polygon, 4 px padding) and
       crops.jsonl. This is OmniDocBench's block-level OCR evaluation, which NVIDIA's model card reports ("crop mode").
pages: all 1651 pages of the v1.6 release -> pages.jsonl with the page's ground-truth plain text in reading order.
       GT = blocks that carry text (titles, paragraphs, headers/footers, captions, footnotes, references, code);
       table bodies, display formulas, figures and abandoned/masked regions are excluded.

Usage: venv-nemo/bin/python -I prep.py
"""
import json
import os

from PIL import Image

D = os.path.dirname(os.path.abspath(__file__))
IMG = f"{D}/dl/omnidocbench/images"
PAGE_CATS = {"title", "text_block", "header", "footer", "page_number", "page_footnote", "figure_caption",
             "figure_footnote", "table_caption", "table_footnote", "reference", "list_group", "code_txt",
             "code_txt_caption", "equation_caption", "equation_explanation"}


def crops():
    os.makedirs(f"{D}/crops", exist_ok=True)
    pages = json.load(open(f"{D}/dl/odb_v15/OmniDocBench.json"))
    n = 0
    with open(f"{D}/crops.jsonl", "w") as out:
        for p in pages:
            info = p["page_info"]
            im = None
            for i, b in enumerate(p["layout_dets"]):
                at = b.get("attribute") or {}
                txt = (b.get("text") or "").strip()
                if "text_language" not in at or not txt or b.get("ignore"):
                    continue
                if im is None:
                    im = Image.open(f"{IMG}/{info['image_path']}").convert("RGB")
                xs, ys = b["poly"][0::2], b["poly"][1::2]
                box = (max(0, int(min(xs)) - 4), max(0, int(min(ys)) - 4),
                       min(im.width, int(max(xs)) + 4), min(im.height, int(max(ys)) + 4))
                if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                    continue
                cid = f"{os.path.splitext(info['image_path'])[0]}__{i}"
                im.crop(box).save(f"{D}/crops/{cid}.png")
                out.write(json.dumps({"id": cid, "page": info["image_path"], "category": b["category_type"],
                                      "text": txt, "language": at.get("text_language"),
                                      "background": at.get("text_background"),
                                      "rotate": (at.get("text_rotate") or "").strip()}, ensure_ascii=False) + "\n")
                n += 1
    print("crops", n)


def page_list():
    pages = json.load(open(f"{D}/dl/omnidocbench/OmniDocBench.json"))
    with open(f"{D}/pages.jsonl", "w") as out:
        for p in pages:
            info = p["page_info"]
            bl = [b for b in p["layout_dets"] if b["category_type"] in PAGE_CATS and not b.get("ignore")
                  and (b.get("text") or "").strip()]
            bl.sort(key=lambda b: (b.get("order") is None, b.get("order") or 0))
            at = info["page_attribute"]
            out.write(json.dumps({"id": os.path.splitext(info["image_path"])[0], "image": info["image_path"],
                                  "text": "\n".join(b["text"].strip() for b in bl),
                                  "language": at.get("language"), "data_source": at.get("data_source"),
                                  "layout": at.get("layout")}, ensure_ascii=False) + "\n")
    print("pages", len(pages))


if __name__ == "__main__":
    crops()
    page_list()
