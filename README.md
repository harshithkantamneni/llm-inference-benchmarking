# LLM Inference Benchmarking — Qwen2.5-7B-Instruct on H100 (vLLM)

Profiling LLM inference serving on a single **NVIDIA H100 NVL**: a latency–throughput
characterization, the impact of **CUDA graphs**, and an **FP8 vs FP16** quantization
comparison — each as a clean controlled experiment with honest methodology and reproducible plots.

**Stack:** Qwen2.5-7B-Instruct · vLLM 0.23.0 · NVIDIA H100 NVL (94 GB) · benchmarked with `vllm bench serve` (p99 TTFT, Poisson arrivals, fixed output length)

---

## Key findings

### 1. Throughput saturates near ~55 req/s · ~7,000 tok/s, and overload shows up in TTFT, not TPOT

As offered load rises, the two latency metrics move very differently:

- **p99 TTFT** is 45-58 ms up to 16 req/s and 102 ms at 32 req/s. An unbounded burst reaches **~1,970 ms**.
- **TPOT** over the same range rises only **~2.3×** (6.5 ms → 15.3 ms).

Once the server saturates, queueing delays the first token while per-token latency stays well-behaved. In this data, that makes TTFT the metric to set an operating rate from. See the full curve in [Results](#results).

**Limits of this data.** Each rate is one run. The request count wasn't logged. Working back from throughput and drain time gives about 200-275 requests per run, so each p99 is roughly the 2nd or 3rd worst request. The burst's TTFT scales with how many requests arrive at once, so it isn't a capacity number. At 64 req/s the offered load is above capacity and the run lasts only about 5 seconds, so that row isn't a steady operating point.

![Latency vs achieved throughput — TTFT cliffs while TPOT degrades gently](plots/sweep_latency.png)

### 2. CUDA graphs cut decode TPOT 31-34%

Comparing the default (CUDA graphs on) against `--enforce-eager` (graphs off):

| Rate (req/s) | TPOT graphs on (ms) | TPOT eager (ms) | Gap (ms) | Reduction (%) |
|---:|---:|---:|---:|---:|
| 4  | 6.20 | 9.37 | 3.17 | 34% |
| 8  | 6.39 | 9.27 | 2.88 | 31% |
| 32 | 7.66 | 11.30 | 3.64 | 32% |

CUDA graphs remove about 3 ms per token, a consistent **31-34%** TPOT reduction. The likely reason is lower launch overhead across the many small kernels of a token step. That wasn't profiled here. The reduction divides the gap by the eager time. The CSV's `gap_pct` column divides by the graphs-on time, which is why it reads 45-51%.

![TPOT: CUDA graphs on vs eager](plots/cuda_graphs_tpot.png)

### 3. FP8 gives 27-36% lower TPOT and ~25% higher peak throughput, quality preserved

Running FP8 (`RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic`) against FP16 on the identical protocol:

- **Decode latency:** FP8 has **27-36% lower** TPOT across the sweep (e.g. offline: 14.06 → 9.31 ms).
- **Peak throughput:** FP8 reaches **8,810 tok/s** offline vs FP16's **7,028 tok/s** — about **+25%**.
- **Quality:** preserved on real-prompt coherence checks (see [METHODOLOGY](METHODOLOGY.md)), so the
  speedup is end-to-end, not just a microbenchmark win.

The speedup lands well below the ~2× you might expect from FP8 matmul alone. Assuming a 2× matmul speedup, Amdahl's law gives a matmul fraction of 54-72% from TPOT and 40% from peak throughput. No single fraction fits, so one Amdahl term doesn't explain this data. The decode step wasn't profiled, so where the rest of the time goes isn't measured here.

![FP8 vs FP16 — decode latency and throughput](plots/fp8_vs_fp16.png)

---

## Results

Full latency–throughput sweep ([`results/vllm_sweep_qwen7b.csv`](results/vllm_sweep_qwen7b.csv)):

| Offered rate (req/s) | Achieved req/s | Output tok/s | p99 TTFT (ms) | TPOT (ms) |
|---:|---:|---:|---:|---:|
| 4   | 3.94  | 503.73  | 45.26   | 6.52  |
| 8   | 7.74  | 991.03  | 53.99   | 6.62  |
| 16  | 14.96 | 1915.51 | 58.33   | 7.07  |
| 32  | 27.80 | 3558.93 | 102.16  | 8.22  |
| 64  | 46.50 | 5951.47 | 144.15  | 11.55 |
| offline (∞) | 54.91 | 7028.38 | 1970.14 | 15.32 |

![Throughput saturation curve](plots/sweep_saturation.png)

FP8 vs FP16 ([`results/fp8_vs_fp16.csv`](results/fp8_vs_fp16.csv)):

| Rate (req/s) | FP16 TPOT (ms) | FP8 TPOT (ms) | FP16 tok/s | FP8 tok/s |
|---:|---:|---:|---:|---:|
| 4   | 6.20  | 4.53 | 503  | 506  |
| 16  | 6.78  | 4.48 | 1916 | 1957 |
| 64  | 10.80 | 6.90 | 5951 | 6645 |
| offline (∞) | 14.06 | 9.31 | 7028 | 8810 |

TPOT differs between this table and the sweep at the same throughput (6.20 vs 6.52 ms at 4 req/s, 14.06 vs 15.32 ms offline), so the two CSVs don't record the same statistic. The repo can't settle which is p99, so TPOT isn't labeled p99 here, even though the CSV column names say so. Comparisons within one table aren't affected.

Raw CSVs for every experiment live in [`results/`](results/); the plots above are regenerated from
them by [`plot_results.py`](plot_results.py).

---

## Reproduce the plots

```bash
python3 -m pip install pandas matplotlib
python3 plot_results.py        # reads results/*.csv, writes PNGs to plots/
```

The benchmark numbers themselves were collected with vLLM's `vllm bench serve` against an
OpenAI-compatible endpoint — see [METHODOLOGY.md](METHODOLOGY.md) for the exact protocol (controlled
single-variable comparisons, p99 TTFT reporting, `--ignore-eos` fixed output lengths, Poisson arrivals).

---

## Repository layout

```
.
├── README.md                  # this file — findings, results, plots
├── METHODOLOGY.md             # experimental protocol and principles
├── plot_results.py            # regenerates plots/ from results/*.csv
├── results/
│   ├── vllm_sweep_qwen7b.csv      # latency–throughput sweep
│   ├── cuda_graphs_experiment.csv # CUDA graphs on vs eager
│   ├── fp8_vs_fp16.csv            # FP8 vs FP16 quantization
│   └── trtllm_install_notes.md    # TensorRT-LLM blocker + next-step plan
└── plots/                     # generated PNGs (committed for convenience)
    ├── sweep_saturation.png
    ├── sweep_latency.png
    ├── cuda_graphs_tpot.png
    └── fp8_vs_fp16.png
```

---

## Known limitation / next step

A **TensorRT-LLM** comparison is the planned next step but is **not yet included**. A direct `pip`
install hit CUDA-version library conflicts — which is exactly why NVIDIA ships TensorRT-LLM as a
pre-built NGC container. The next iteration re-runs the FP16/FP8 sweeps inside that container on the
same H100 with the identical protocol. Details: [`results/trtllm_install_notes.md`](results/trtllm_install_notes.md).

## Hardware & software

| | |
|---|---|
| GPU | NVIDIA H100 NVL (94 GB) |
| Model | Qwen2.5-7B-Instruct (FP8: `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic`) |
| Serving engine | vLLM 0.23.0 |
| Benchmark tool | `vllm bench serve` (OpenAI-compatible endpoint) |
| Metrics | p99 TTFT, TPOT, throughput (req/s and output tok/s) |

## License

Released under the [MIT License](LICENSE).
