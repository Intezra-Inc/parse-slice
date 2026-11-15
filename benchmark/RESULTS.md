# OCR benchmark, full run on two machines (OmniDocBench)

## Full pages: accuracy (1,645 pages)

| engine | machine | words found | precision | EN | ZH | reading-order error | failed pages |
|---|---|---|---|---|---|---|---|
| Nemotron OCR v2, multilingual | H100 NVL | **0.784** | 0.729 | 0.81 | 0.767 | 0.521 | 0 |
| Nemotron OCR v2, multilingual | RTX PRO 6000 Blackwell Max-Q | **0.784** | 0.729 | 0.81 | 0.766 | 0.521 | 0 |
| Nemotron OCR v2, English | H100 NVL | **0.441** | 0.504 | 0.847 | 0.058 | 0.702 | 0 |
| Nemotron OCR v2, English | RTX PRO 6000 Blackwell Max-Q | **0.441** | 0.504 | 0.847 | 0.058 | 0.703 | 0 |
| PP-OCRv5 server | H100 NVL | **0.845** | 0.737 | 0.796 | 0.893 | 0.521 | 0 |
| PP-OCRv5 server | RTX PRO 6000 Blackwell Max-Q | **0.845** | 0.737 | 0.796 | 0.892 | 0.521 | 0 |
| PP-OCRv6 medium | H100 NVL | **0.913** | 0.758 | 0.906 | 0.92 | 0.511 | 0 |
| PP-OCRv6 medium | RTX PRO 6000 Blackwell Max-Q | **0.913** | 0.757 | 0.906 | 0.92 | 0.511 | 0 |
| PaddleOCR-VL-1.6 | H100 NVL | **0.949** | 0.775 | 0.953 | 0.944 | 0.3 | 0 |
| PaddleOCR-VL-1.6 | RTX PRO 6000 Blackwell Max-Q | **0.949** | 0.775 | 0.953 | 0.944 | 0.3 | 0 |
| Nemotron 3.5 Parse | H100 NVL | **0.895** | 0.72 | 0.925 | 0.866 | 0.361 | 50 |
| Nemotron 3.5 Parse | RTX PRO 6000 Blackwell Max-Q | **0.897** | 0.721 | 0.924 | 0.87 | 0.359 | 48 |
| Nemotron 3.5 Parse, double pass | H100 NVL | **0.895** | 0.695 | 0.926 | 0.865 | 0.38 | 52 |
| Nemotron 3.5 Parse, double pass | RTX PRO 6000 Blackwell Max-Q | **0.897** | 0.698 | 0.924 | 0.87 | 0.379 | 50 |
| Nemotron 3.5 Parse + slicing (parse-slice) | H100 NVL | **0.924** | 0.747 | 0.938 | 0.914 | 0.342 | 0 |
| Nemotron 3.5 Parse + slicing (parse-slice) | RTX PRO 6000 Blackwell Max-Q | **0.924** | 0.747 | 0.939 | 0.914 | 0.342 | 0 |
| Nemotron 3.5 Super VL (2 GPUs, NVFP4) | (VLM run, greedy) | **0.913** | 0.74 | 0.937 | 0.89 | 0.357 | 0 |
| Nemotron 3.5 Super VL (2 GPUs, NVFP4), looping pages resampled (63; 35 still capped) | (VLM run) | **0.919** | 0.75 | 0.941 | 0.897 | 0.35 | 0 |
| Gemma 4 31B (BF16) | (VLM run, greedy) | **0.87** | 0.708 | 0.948 | 0.815 | 0.411 | 0 |
| Gemma 4 31B (BF16), looping pages resampled (189; 75 still capped) | (VLM run) | **0.922** | 0.71 | 0.954 | 0.885 | 0.374 | 0 |
| Gemma 4 26B-A4B (BF16) | (VLM run, greedy) | **0.834** | 0.686 | 0.923 | 0.782 | 0.46 | 0 |
| Gemma 4 26B-A4B (BF16), looping pages resampled (274; 95 still capped) | (VLM run) | **0.907** | 0.7 | 0.945 | 0.864 | 0.405 | 0 |

## Text blocks (23,078, OmniDocBench v1.5), edit distance (lower is better)

| engine | machine | all | English |
|---|---|---|---|
| Nemotron OCR v2, multilingual | H100 NVL | 0.077 | 0.057 |
| Nemotron OCR v2, multilingual | RTX PRO 6000 Blackwell Max-Q | 0.078 | 0.057 |
| Nemotron OCR v2, English | H100 NVL | 0.392 | 0.05 |
| Nemotron OCR v2, English | RTX PRO 6000 Blackwell Max-Q | 0.391 | 0.05 |
| PP-OCRv5 server | H100 NVL | 0.053 | 0.043 |
| PP-OCRv5 server | RTX PRO 6000 Blackwell Max-Q | 0.052 | 0.043 |
| PP-OCRv6 medium | H100 NVL | 0.045 | 0.034 |
| PP-OCRv6 medium | RTX PRO 6000 Blackwell Max-Q | 0.045 | 0.035 |

## Speed, one page at a time, one GPU (pages/s; first 5 pages dropped)

| engine | H100, first 300 | Blackwell, first 300 | H100, all pages | Blackwell, all pages | Blackwell / H100 (all) |
|---|---|---|---|---|---|
| Nemotron OCR v2, multilingual | 18.79 | 17.6 | 20.25 | 19.39 | 0.96 |
| Nemotron OCR v2, English | 20.92 | 21.63 | 21.98 | 22.63 | 1.03 |
| PP-OCRv5 server | 1.9 | 1.94 | 1.5 | 1.55 | 1.03 |
| PP-OCRv6 medium | 2.12 | 1.99 | 1.69 | 1.58 | 0.93 |
| PaddleOCR-VL-1.6 | 0.68 | 0.61 | 0.76 | 0.63 | 0.83 |
| Nemotron 3.5 Parse | 0.64 | 0.5 | 0.53 | 0.47 | 0.89 |
| Nemotron 3.5 Parse, double pass | 0.47 | 0.41 | 0.45 | 0.39 | 0.87 |
| Nemotron 3.5 Parse + slicing (parse-slice) | 0.64 | 0.47 | 0.48 | 0.37 | 0.77 |

## General VLMs: throughput (concurrent, 32 requests in flight)

| model | GPUs | pages/s | pages/s per GPU | pages hitting the 16,384-token cap |
|---|---|---|---|---|
| Nemotron 3.5 Super VL (2 GPUs, NVFP4) | 2 | 0.528 | 0.264 | 63 |
| Gemma 4 31B (BF16) | 1 | 0.133 | 0.133 | 189 |
| Gemma 4 26B-A4B (BF16) | 1 | 0.333 | 0.333 | 274 |

## Reproducibility: H100 vs Blackwell (same software, same pages)

| engine | pages with identical text | words found, H100 - Blackwell |
|---|---|---|
| Nemotron OCR v2, multilingual | 469/1651 | +0.0 |
| Nemotron OCR v2, English | 557/1651 | +0.0 |
| PP-OCRv5 server | 1114/1651 | +0.0 |
| PP-OCRv6 medium | 1430/1651 | +0.0 |
| PaddleOCR-VL-1.6 | 1231/1651 | +0.0 |
| Nemotron 3.5 Parse | 972/1651 | -0.002 |
| Nemotron 3.5 Parse, double pass | 888/1651 | -0.002 |
| Nemotron 3.5 Parse + slicing (parse-slice) | 938/1651 | +0.0 |
