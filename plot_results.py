#!/usr/bin/env python3
"""Generate benchmark plots from results/*.csv.

Every number plotted is read straight from results/*.csv — nothing is
hard-coded here. Writes publication-style PNGs to plots/.

Usage:
    python3 -m pip install pandas matplotlib
    python3 plot_results.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

# Consistent palette across figures.
C_PRIMARY = "#1f77b4"   # "good" / optimized config (graphs on, FP8, throughput)
C_SECONDARY = "#d62728"  # "baseline" / worse config (eager, FP16, latency)
C_GREEN = "#2ca02c"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _rate_labels(rates):
    """Render the 'rate' column as tick labels, mapping inf -> 'offline'."""
    out = []
    for r in rates:
        s = str(r).strip().lower()
        out.append("offline\n(unbounded)" if s in ("inf", "infinity") else str(int(float(r))))
    return out


def plot_sweep_saturation(df):
    labels = _rate_labels(df["rate"])
    x = range(len(df))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(x, df["throughput_toks"], marker="o", color=C_PRIMARY, lw=2, label="Output tok/s")
    ax1.set_xlabel("Offered request rate (req/s)")
    ax1.set_ylabel("Throughput (output tok/s)", color=C_PRIMARY)
    ax1.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.plot(x, df["throughput_reqs"], marker="s", color=C_GREEN, lw=2, label="Completed req/s")
    ax2.set_ylabel("Throughput (req/s)", color=C_GREEN)
    ax2.tick_params(axis="y", labelcolor=C_GREEN)
    ax2.grid(False)

    sat_tok = df["throughput_toks"].iloc[-1]
    sat_req = df["throughput_reqs"].iloc[-1]
    ax1.annotate(
        f"saturation\n~{sat_req:.0f} req/s · ~{sat_tok:,.0f} tok/s",
        xy=(len(df) - 1, sat_tok), xytext=(len(df) - 2.6, sat_tok * 0.72),
        fontsize=9, color="#333",
        arrowprops=dict(arrowstyle="->", color="#666"),
    )
    ax1.set_title("Throughput saturates as offered load increases")
    fig.tight_layout()
    out = PLOTS / "sweep_saturation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_sweep_latency(df):
    xt = df["throughput_toks"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    axA.plot(xt, df["p99_ttft_ms"], marker="o", color=C_SECONDARY, lw=2)
    axA.set_yscale("log")
    axA.set_xlabel("Achieved throughput (output tok/s)")
    axA.set_ylabel("p99 TTFT (ms, log scale)")
    axA.set_title("Time-to-first-token cliffs at saturation")
    lo, hi = df["p99_ttft_ms"].iloc[0], df["p99_ttft_ms"].iloc[-1]
    axA.annotate(
        f"~{lo:.0f} ms  ->  {hi:,.0f} ms\n(~{hi / lo:.0f}x)",
        xy=(xt.iloc[-1], hi), xytext=(xt.iloc[1], hi * 0.35),
        fontsize=9, color="#333",
        arrowprops=dict(arrowstyle="->", color="#666"),
    )

    axB.plot(xt, df["p99_tpot_ms"], marker="o", color=C_PRIMARY, lw=2)
    axB.set_xlabel("Achieved throughput (output tok/s)")
    axB.set_ylabel("p99 TPOT (ms)")
    axB.set_title("Time-per-output-token degrades gently")
    lo2, hi2 = df["p99_tpot_ms"].iloc[0], df["p99_tpot_ms"].iloc[-1]
    axB.annotate(
        f"~{lo2:.1f} -> {hi2:.1f} ms (~{hi2 / lo2:.1f}x)",
        xy=(xt.iloc[-1], hi2), xytext=(xt.iloc[0], hi2 * 0.9),
        fontsize=9, color="#333",
    )

    fig.suptitle("Latency vs. achieved throughput: TTFT and TPOT degrade asymmetrically", y=1.02)
    fig.tight_layout()
    out = PLOTS / "sweep_latency.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cuda_graphs(df):
    x = range(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar([i - w / 2 for i in x], df["tpot_eager_ms"], width=w,
           color=C_SECONDARY, label="Eager (CUDA graphs off)")
    ax.bar([i + w / 2 for i in x], df["tpot_graphs_on_ms"], width=w,
           color=C_PRIMARY, label="CUDA graphs on (default)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{int(r)} req/s" for r in df["rate"]])
    ax.set_ylabel("p99 TPOT (ms)")
    ax.set_title("CUDA graphs amortize per-step launch overhead")
    ax.legend(frameon=False)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, max(row["tpot_eager_ms"], row["tpot_graphs_on_ms"]) + 0.25,
                f"-{int(row['gap_pct'])}%", ha="center", fontsize=9, color="#333")
    fig.tight_layout()
    out = PLOTS / "cuda_graphs_tpot.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_fp8_vs_fp16(df):
    labels = _rate_labels(df["rate"])
    x = range(len(df))
    w = 0.38
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    axA.bar([i - w / 2 for i in x], df["fp16_tpot_ms"], width=w, color=C_SECONDARY, label="FP16")
    axA.bar([i + w / 2 for i in x], df["fp8_tpot_ms"], width=w, color=C_PRIMARY, label="FP8")
    axA.set_xticks(list(x))
    axA.set_xticklabels(labels)
    axA.set_xlabel("Offered request rate (req/s)")
    axA.set_ylabel("p99 TPOT (ms)")
    axA.set_title("Decode latency: FP8 vs FP16")
    axA.legend(frameon=False)

    axB.bar([i - w / 2 for i in x], df["fp16_tput"], width=w, color=C_SECONDARY, label="FP16")
    axB.bar([i + w / 2 for i in x], df["fp8_tput"], width=w, color=C_PRIMARY, label="FP8")
    axB.set_xticks(list(x))
    axB.set_xticklabels(labels)
    axB.set_xlabel("Offered request rate (req/s)")
    axB.set_ylabel("Throughput (output tok/s)")
    axB.set_title("Throughput: FP8 vs FP16")
    axB.legend(frameon=False)

    fig.suptitle("FP8 quantization: faster decode and higher peak throughput", y=1.02)
    fig.tight_layout()
    out = PLOTS / "fp8_vs_fp16.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    PLOTS.mkdir(exist_ok=True)
    sweep = pd.read_csv(RESULTS / "vllm_sweep_qwen7b.csv", dtype={"rate": str})
    graphs = pd.read_csv(RESULTS / "cuda_graphs_experiment.csv")
    quant = pd.read_csv(RESULTS / "fp8_vs_fp16.csv", dtype={"rate": str})

    written = [
        plot_sweep_saturation(sweep),
        plot_sweep_latency(sweep),
        plot_cuda_graphs(graphs),
        plot_fp8_vs_fp16(quant),
    ]
    print("Wrote:")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
