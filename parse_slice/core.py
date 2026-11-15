"""Page slicing around NVIDIA's Nemotron 3.5 Parse client.

Nemotron 3.5 Parse reads a whole page in one request. Its context is 16,384 tokens, of which the page image takes
about 7,600, so NVIDIA's client caps the output at 8,744 tokens. A dense page (a newspaper, a long reference list)
can need more than that, and the request ends with finish_reason="length". The client then raises GenerationError and
the page comes back empty.

SlicingParser parses the page as usual. Only when the full-page request runs out of room does it cut the page in two
along the emptiest band of rows near the middle (so the cut runs between text lines, not through them), parse each
half the same way, and join the halves in reading order (top, then bottom). A half that still runs out of room is cut
again, up to max_depth levels (2 ** max_depth pieces). Pages that fit are parsed exactly as NVIDIA's client parses
them, with one request.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from nemotron_parse.inference import DocumentParser, GenerationError

__all__ = ["SlicingParser", "SliceResult", "Piece"]


@dataclass
class Piece:
    box: tuple[int, int, int, int]  # (left, top, right, bottom) in page pixels
    depth: int
    ok: bool
    error: str | None = None


@dataclass
class SliceResult:
    markdown: str
    pieces: list[Piece] = field(default_factory=list)
    requests: int = 0
    seconds: float = 0.0

    @property
    def sliced(self) -> bool:
        return len(self.pieces) > 1

    @property
    def complete(self) -> bool:
        return all(p.ok for p in self.pieces)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(sliced=self.sliced, complete=self.complete)
        return d


def _is_length_error(exc: BaseException) -> bool:
    return isinstance(exc, GenerationError) and (exc.response or {}).get("finish_reason") == "length"


def find_cut(img: Image.Image, vertical: bool = False, window: float = 0.15) -> int:
    """Position of the emptiest band of rows (or columns) within +-window of the middle.

    Ink per row = mean darkness of the grayscale image; smoothed over ~1% of the size so a single light row inside a
    line of text doesn't win. Falls back to the exact middle on blank or uniform images.
    """
    a = np.asarray(img.convert("L"), dtype=np.float32)
    ink = 255.0 - (a.mean(axis=0) if vertical else a.mean(axis=1))
    n = ink.size
    lo, hi = int(n * (0.5 - window)), int(n * (0.5 + window))
    if hi - lo < 3:
        return n // 2
    k = max(3, n // 100) | 1
    smooth = np.convolve(ink, np.ones(k) / k, mode="same")[lo:hi]
    if float(smooth.max() - smooth.min()) < 1e-3:
        return n // 2
    best = np.flatnonzero(smooth <= smooth.min() + 1e-6)
    return lo + int(best[len(best) // 2])  # middle of the emptiest band


class SlicingParser:
    """Nemotron 3.5 Parse with automatic page slicing for pages that run out of output room.

    base_url / model:  the vLLM server running Nemotron 3.5 Parse (OpenAI-compatible API).
    mode:              NVIDIA's parse mode, "single" (default, one request per page) or "double" (adds table/chart
                       crop passes). Pieces are parsed in the same mode.
    max_depth:         how many times a piece may be cut again (default 3, i.e. at most 8 pieces per page).
    min_piece_px:      never cut a piece shorter (or narrower) than this.
    Any other keyword arguments go to nemotron_parse.inference.DocumentParser.
    Thread-safe: one SlicingParser can serve many threads (DocumentParser shares one httpx client).
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "nemotron-parse", *,
                 mode: str = "single", max_depth: int = 3, min_piece_px: int = 256, **parser_kwargs):
        if mode not in ("single", "double"):
            raise ValueError("mode must be single or double")
        self.mode, self.max_depth, self.min_piece_px = mode, max_depth, min_piece_px
        self.parser = DocumentParser(base_url=base_url, model=model, **parser_kwargs)

    def close(self) -> None:
        self.parser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def parse(self, image) -> SliceResult:
        """Parse one page (a path or a PIL image). Returns the Markdown and how the page was cut."""
        t = time.perf_counter()
        if not isinstance(image, Image.Image):
            with Image.open(image) as src:
                image = src.convert("RGB")
        else:
            image = image.convert("RGB")
        res = SliceResult(markdown="")
        parts = self._parse_piece(image, (0, 0, image.width, image.height), 0, res)
        res.markdown = "\n\n".join(p for p in parts if p.strip())
        res.seconds = round(time.perf_counter() - t, 3)
        return res

    def _parse_piece(self, page: Image.Image, box, depth: int, res: SliceResult) -> list[str]:
        crop = page if box == (0, 0, page.width, page.height) else page.crop(box)
        res.requests += 1
        try:
            md = self.parser.parse(crop, mode=self.mode)["markdown"]
            res.pieces.append(Piece(box, depth, True))
            return [md]
        except Exception as exc:  # noqa: BLE001 - classified below
            if not _is_length_error(exc):
                res.pieces.append(Piece(box, depth, False, repr(exc)[:300]))
                return [""]
            if getattr(exc, "stage", "fullpage") != "fullpage" and self.mode == "double":
                # a table/chart crop ran out of room: keep the page text by re-reading this piece in single mode
                try:
                    res.requests += 1
                    md = self.parser.parse(crop, mode="single")["markdown"]
                    res.pieces.append(Piece(box, depth, True, "crop pass truncated; single-mode text kept"))
                    return [md]
                except Exception as exc2:  # noqa: BLE001
                    if not _is_length_error(exc2):
                        res.pieces.append(Piece(box, depth, False, repr(exc2)[:300]))
                        return [""]
            x0, y0, x1, y1 = box
            w, h = x1 - x0, y1 - y0
            vertical = w > 1.3 * h  # wide pieces (e.g. a two-page spread) are cut into left and right
            if depth >= self.max_depth or (w if vertical else h) < 2 * self.min_piece_px:
                res.pieces.append(Piece(box, depth, False, "output limit reached; piece too small or too deep to cut"))
                return [""]
            c = find_cut(crop, vertical=vertical)
            c = min(max(c, self.min_piece_px), (w if vertical else h) - self.min_piece_px)
            a, b = ((x0, y0, x0 + c, y1), (x0 + c, y0, x1, y1)) if vertical else ((x0, y0, x1, y0 + c), (x0, y0 + c, x1, y1))
            return self._parse_piece(page, a, depth + 1, res) + self._parse_piece(page, b, depth + 1, res)
