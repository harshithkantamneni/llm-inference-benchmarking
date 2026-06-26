# TensorRT-LLM — install notes & status

**Status: planned, not yet run.** This note documents why the TensorRT-LLM comparison
is deferred, so the gap is transparent and reproducible.

## What happened
Installing TensorRT-LLM directly with `pip` into the existing benchmarking environment
failed with CUDA-version / shared-library conflicts: the TensorRT and CUDA runtime
libraries the wheel expects did not line up with the CUDA toolkit already present in the
environment. This is the common "the build only works against one specific CUDA/TensorRT
version" class of failure.

## Why this is expected
NVIDIA ships TensorRT-LLM primarily as a **pre-built NGC container** precisely because the
engine is tightly pinned to specific CUDA / TensorRT / driver versions. A
`pip`-into-an-existing-environment install is brittle for exactly this reason, so the
conflict above is the expected outcome rather than a one-off.

## Planned next step
Re-run the FP16 and FP8 sweeps **inside the official TensorRT-LLM NGC container** on the same
H100 NVL, using the identical protocol (model, sequence lengths, `--ignore-eos`, Poisson
arrivals, p99 reporting), then add a vLLM-vs-TensorRT-LLM section to the results.
