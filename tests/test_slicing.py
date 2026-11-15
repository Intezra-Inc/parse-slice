"""Offline tests: a fake backend stands in for the server and 'runs out of room' when a piece is taller than LIMIT."""
import numpy as np
from PIL import Image, ImageDraw

from parse_slice import SlicingParser, find_cut

LIMIT = 700


def page(h=2000, w=1000, gap_at=1100):
    img = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(img)
    for y in range(40, h - 40, 30):
        if abs(y - gap_at) > 60:
            d.rectangle([60, y, w - 60, y + 14], fill="black")  # a "text line" every 30 px, blank band at gap_at
    return img


def fake_backend(image, *, prompt, stage, settings):
    if image.height > LIMIT:
        return {"raw_out": "", "finish_reason": "length", "usage": {}, "error": None}
    return {"raw_out": f"<x_0.0><y_0.0>piece h={image.height}<x_1.0><y_1.0><class_Text>", "finish_reason": "stop",
            "usage": {}, "error": None}


def test_find_cut_lands_in_blank_band():
    c = find_cut(page(gap_at=1100))
    assert 1060 <= c <= 1140, c


def test_find_cut_uniform_image_is_middle():
    assert find_cut(Image.new("RGB", (100, 400), "white")) == 200


def test_page_that_fits_is_one_request():
    with SlicingParser(backend=fake_backend) as sp:
        r = sp.parse(page(h=600))
    assert r.requests == 1 and not r.sliced and r.complete


def test_dense_page_is_cut_until_pieces_fit():
    with SlicingParser(backend=fake_backend) as sp:
        r = sp.parse(page(h=2000))
    assert r.sliced and r.complete
    assert all(p.box[3] - p.box[1] <= LIMIT for p in r.pieces if p.ok)
    tops = [p.box[1] for p in r.pieces if p.ok]
    assert tops == sorted(tops)  # reading order: top to bottom
    assert r.markdown.count("piece h=") == len(tops)


def test_depth_limit_reports_incomplete():
    with SlicingParser(backend=fake_backend, max_depth=1) as sp:
        r = sp.parse(page(h=3000))
    assert not r.complete and any("output limit" in (p.error or "") for p in r.pieces)


def test_no_slice_matches_nvidia_behaviour():
    with SlicingParser(backend=fake_backend, max_depth=0) as sp:
        r = sp.parse(page(h=2000))
    assert r.requests == 1 and not r.complete and r.markdown == ""
