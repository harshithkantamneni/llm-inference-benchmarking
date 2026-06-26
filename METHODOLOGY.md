# Benchmarking Methodology

All benchmarks run on a single NVIDIA H100 NVL (94GB), serving Qwen2.5-7B-Instruct via vLLM 0.23.0, using vLLM's `vllm bench serve` against an OpenAI-compatible endpoint.

## Principles
- Controlled comparison: only one variable changes per experiment; model, precision, dataset, and lengths are held constant otherwise.
- Percentile reporting (p99), not means, since tail latency is the real serving constraint.
- Fixed output length via `--ignore-eos` for clean, comparable measurements.
- Poisson request arrival (the tool's default) to model realistic load.

## Server
FP8 runs used the pre-quantized `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic`.

## Latency-throughput sweep
Swept request rate from light load to saturation — 4, 8, 16, 32, and 64 req/s plus an offline (unbounded) run — to map the latency-throughput curve and locate the SLO knee. p99 TTFT and p99 TPOT are tracked separately because, under queueing, they degrade on very different timescales: time-to-first-token is dominated by admission/queueing delay as the server approaches saturation, while time-per-output-token reflects steady-state decode.

## CUDA graphs experiment
Compared default (graphs on) vs `--enforce-eager` (graphs off) at low-to-moderate rates, where per-step launch overhead is a larger fraction of step time.

## Quantization experiment
Benchmarked FP8 vs FP16 on the identical protocol, measuring both performance (throughput, p99 TPOT) and output quality (coherence on real prompts), to validate quantization helped end-to-end, not just in microbenchmarks.

## Profiling
Captured torch-profiler traces during a small request burst (`--profile`) to inspect the per-token kernel launch pattern and confirm decode is dominated by memory-bound projection matmuls.

## Known limitation
TensorRT-LLM comparison was attempted but not completed; the pip install hit CUDA-version library conflicts (documented in results/trtllm_install_notes.md), which is why NVIDIA ships TRT-LLM as a pre-built container. This is the planned next step via the NGC container.
