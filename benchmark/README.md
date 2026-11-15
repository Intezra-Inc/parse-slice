# Benchmark

Scripts behind `RESULTS.md`: OmniDocBench, all 1,651 full pages (v1.6) and 23,078 text blocks (v1.5), on one
H100 NVL and one RTX PRO 6000 Blackwell Max-Q, one engine at a time.

## Layout this expects

Everything runs from this directory:

```
dl/omnidocbench/           OmniDocBench v1.6 (images/ + OmniDocBench.json), from Hugging Face: opendatalab/OmniDocBench
dl/odb_v15/OmniDocBench.json   the v1.5 annotations (text-block language/background/rotation tags)
dl/nemotron-3.5-parse/     nvidia/Nemotron-3.5-Parse
dl/nemotron-ocr-v2/        nvidia/nemotron-ocr-v2
dl/paddleocr-vl-1.6/       PaddleOCR-VL-1.6 weights (PaddleOCR downloads the PP-OCR models itself)
build/nemotron-ocr/        a copy of dl/nemotron-ocr-v2/nemotron-ocr with -std=c++17 changed to -std=c++20 in
                           scripts/build-extension.py (needed with PyTorch 2.14)
build/parse35/             a copy of the Python package files from dl/nemotron-3.5-parse
```

## Steps

```bash
bash env/setup_envs.sh                      # venv-nemo, venv-paddle, venv-vllm with the exact versions in env/*.txt
venv-nemo/bin/python -I prep.py             # pages.jsonl, crops.jsonl and crops/ (the cut-out text blocks)
./run_full.sh h100 0                        # every engine on GPU 0; writes preds_full/h100/
venv-nemo/bin/python -I score_full.py       # RESULTS.md tables
```

`run_full.sh` runs Nemotron OCR v2 (multilingual and English), PP-OCRv5, PP-OCRv6, PaddleOCR-VL-1.6 and Nemotron 3.5
Parse in single, double and slicing modes, recording the time for every page. `run_slice.py` uses the parse-slice
package from the repository root. `run_vlm.py` and `serve_gemma.sh` cover the general vision models (Gemma 4,
Nemotron 3.5 Super VL); they run concurrently, so they report throughput rather than per-page latency.

## Scoring

- Full pages: the reference is every text block in reading order, without table bodies, display formulas or
  figures. "Words found" is the share of reference words in the output (Chinese per character); the reading-order
  error is the normalised edit distance in reading order.
- Text blocks: normalised edit distance after OmniDocBench's own clean-up, averaged per block, empty outputs skipped.
- Markdown from the vision models is reduced to plain text with the same rules for every engine.
- Speed: pages / total seconds, first 5 pages dropped.
