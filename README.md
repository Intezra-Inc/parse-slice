# parse-slice

Document-to-Markdown with NVIDIA's [Nemotron 3.5 Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse), plus
automatic page slicing for pages too dense to parse in one go.

Nemotron 3.5 Parse reads a whole page per request. Its context is 16,384 tokens and the page image takes about 7,600
of them, so the output is capped at 8,744 tokens. A dense page, such as a newspaper or a long reference list, can need
more. The request then stops with `finish_reason="length"` and NVIDIA's client returns nothing for that page. On
OmniDocBench that happens on 3% of pages.

parse-slice parses every page exactly as NVIDIA's client does. Only when a page runs out of room does it cut the
page in two along the emptiest band of rows near the middle, so the cut runs between lines of text, then parse each
half and join them top to bottom. A half that still doesn't fit is cut again, up to 8 pieces per page.

| OmniDocBench v1.6, 1,645 pages, H100 NVL | Words found | English | Chinese | Reading-order error (lower is better) | Failed pages |
|---|---|---|---|---|---|
| Nemotron 3.5 Parse, NVIDIA's client | 89.5% | 92.5% | 86.6% | 0.361 | 50 |
| Nemotron 3.5 Parse + parse-slice | **92.4%** | **93.8%** | **91.4%** | **0.342** | **0** |

parse-slice cut 50 of the 1,651 pages (into 104 pieces), so the run took 1,759 requests instead of 1,651. Results on
an RTX PRO 6000 Blackwell are the same to within 0.2 points. Details and other engines: `benchmark/RESULTS.md`.

## Quick start (Docker)

Needs one NVIDIA GPU, Docker with the NVIDIA Container Toolkit, and about 30 GB of disk. Tested on H100 NVL and
RTX PRO 6000 Blackwell Max-Q, with vLLM 0.31.0 and 0.32.0 (same accuracy and speed). The server reserves 90% of the GPU's
memory by default, as in NVIDIA's setup; the model itself needs only a few GB.

```bash
git clone https://github.com/Intezra-Inc/parse-slice.git && cd parse-slice
./scripts/fetch_model.sh                       # NVIDIA's model repository into ./model (~2.4 GB)
docker compose build                           # vLLM 0.32.0 + Nemotron 3.5 Parse + parse-slice
docker compose up -d parse                     # start the server; first start takes a few minutes
mkdir -p input output && cp your-files/* input/
docker compose run --rm --user "$(id -u):$(id -g)" slice /input -o /output
```

`output/` then holds one `.md` file per image or PDF, and `results.jsonl` with one line per page: whether it was
sliced, how many requests it took, the time, and any piece that still failed. Running the same command again skips
files that are already done. `--user` makes the output files yours; without it they belong to root.

Options: `PARSE_GPU=1` picks the GPU and `PARSE_PORT=8001` the host port (the server only listens on localhost).
`docker compose run --rm slice --help` lists the client options.

## Without Docker

```bash
python3 -m venv .venv && . .venv/bin/activate
./scripts/fetch_model.sh
pip install -c docker/constraints.txt "./model[vllm,client]" ".[pdf]"
vllm serve ./model --served-model-name nemotron-parse --trust-remote-code --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs 192 --max-num-batched-tokens 16384 --no-enable-prefix-caching \
  --attention-config '{"flash_attn_version":3}' --limit-mm-per-prompt '{"image":1}' \
  --chat-template ./model/chat_template.jinja --port 8000 &
parse-slice input/ -o output/
```

These are NVIDIA's model-card serving flags. Only the client needs `parse-slice`; it can run on another machine
with `--url http://<server>:8000/v1`.

## From Python

```python
from parse_slice import SlicingParser

with SlicingParser("http://localhost:8000/v1") as sp:
    r = sp.parse("page.png")          # a path or a PIL image
    print(r.markdown)
    print(r.sliced, r.requests, [p.box for p in r.pieces])
```

`SlicingParser` is safe to share between threads. For throughput, keep 32 to 64 pages in flight per GPU, as the CLI
does by default (`--concurrency 64`).

## Options

| CLI flag | Default | What it does |
|---|---|---|
| `--mode single\|double` | `single` | NVIDIA's parse mode. `double` re-reads tables and charts with extra requests. |
| `--concurrency N` | 64 | Pages in flight. |
| `--max-depth N` | 3 | How many times a piece can be cut again (2^N pieces at most). |
| `--no-slice` | off | NVIDIA's behaviour: pages that run out of room stay empty. |
| `--pdf-dpi N` | 200 | Render resolution for PDF pages. |
| `--max-megapixels N` | 400 | Largest input image accepted (Pillow's own limit is ~179). |

## How the cut is chosen

For each piece that runs out of room, parse-slice computes how much ink each row has, smooths it over about 1% of the
height, and cuts at the emptiest band within 15% of the middle. Pieces wider than 1.3 times their height, such as a
scanned two-page spread, are cut into left and right halves instead. Pieces are never cut below 256 px. Text that
spans the cut (for example a column that continues below it) is read in two parts and joined in order, so reading
order within multi-column pages can differ slightly from a single pass.

## Benchmark

`benchmark/` holds the scripts and results for the comparison above: OmniDocBench v1.6, all 1,651 pages, one
H100 NVL and one RTX PRO 6000 Blackwell.

## Licences

The code in this repository is under the Apache License 2.0 (`LICENSE`). It doesn't include any NVIDIA files.
`scripts/fetch_model.sh` downloads NVIDIA's model repository (weights, the `nemotron_parse` package and its vLLM
plugin), which NVIDIA licenses under OpenMDW-1.1; the tokenizer is under MIT. See `model/LICENSE` after downloading.
