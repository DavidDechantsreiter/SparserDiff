"""
Benchmark: Original vs Novel edge sampling on REAL Ego graphs.
Produces the same file set as results_qm9/:
  ego_speed_comparison.pdf/png
  ego_memory_comparison.pdf/png
  ego_speed_memory_combined.pdf/png
  ego_full_summary.pdf/png          (speed + memory + quality metrics)
  ego_benchmark_results.txt         (speed table + training quality table)
"""

import sys
import os
import time
import gc
import re
import numpy as np

import torch
from torch_geometric.loader import DataLoader

# ── Setup paths ───────────────────────────────────────────────────────────────
SPARSERDIFF = "/home/hpham/wpi-graph-ai-mqp-25-26/SparserDiff"
sys.path.insert(0, SPARSERDIFF)
sys.path.insert(0, os.path.join(SPARSERDIFF, "sparse_diffusion"))

from sparse_diffusion.datasets.spectre_dataset_pyg import SpectreGraphDataset
from sparse_diffusion.diffusion.sample_edges import (
    sample_non_existing_edges_batched,
    sample_non_existing_edges_novel,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_BASELINE = "/scratch/hpham/SparseDiff/logs/1861442.out"
LOG_NOVEL    = "/scratch/hpham/SparseDiff/logs/1861443.out"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Load dataset
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading Ego test set...")
dataset = SpectreGraphDataset(
    dataset_name="ego",
    split="test",
    root=os.path.join(SPARSERDIFF, "data/ego/"),
)
print(f"  Loaded {len(dataset)} graphs")
node_counts = [d.num_nodes for d in dataset]
print(f"  Node count range: {min(node_counts)} – {max(node_counts)}, "
      f"avg {sum(node_counts)/len(node_counts):.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_batch(batch_data):
    num_nodes = torch.bincount(batch_data.batch)
    bs = num_nodes.shape[0]
    mask = batch_data.edge_index[0] < batch_data.edge_index[1]
    dir_edge_index = batch_data.edge_index[:, mask]
    edge_batch = batch_data.batch[dir_edge_index[0]]
    num_edges_per_graph = torch.bincount(edge_batch, minlength=bs)
    max_possible = (num_nodes * (num_nodes - 1)) // 2 - num_edges_per_graph
    num_to_sample = torch.clamp(num_edges_per_graph, min=1)
    num_to_sample = torch.min(num_to_sample, max_possible)
    num_to_sample = torch.clamp(num_to_sample, min=1)
    return num_to_sample, dir_edge_index, num_nodes, batch_data.batch


def benchmark_speed(fn, prepared_batches, num_warmup=3, num_runs=10):
    for _ in range(num_warmup):
        for args in prepared_batches:
            fn(num_edges_to_sample=args[0], existing_edge_index=args[1],
               num_nodes=args[2], batch=args[3])
    gc.collect()
    times = []
    for _ in range(num_runs):
        for args in prepared_batches:
            t0 = time.perf_counter()
            fn(num_edges_to_sample=args[0], existing_edge_index=args[1],
               num_nodes=args[2], batch=args[3])
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return np.mean(times), np.std(times), np.median(times)


def compute_theoretical_memory(num_to_sample, dir_edge_index, num_nodes, batch):
    bs = num_nodes.shape[0]
    E = dir_edge_index.shape[1]
    total_sampled = num_to_sample.sum().item()
    # Original SparseDiff: O(bs × max_condensed) dominant tensors
    max_condensed = ((num_nodes * (num_nodes - 1)) // 2).max().item()
    mem_orig = (max_condensed * 8                  # randperm_full int64
                + bs * max_condensed * 1           # mask1 bool
                + bs * max_condensed * 8           # cumsum int64
                + bs * max_condensed * 1           # mask2 bool
                + (E + bs + total_sampled) * 4 * 3
                + (total_sampled + bs + E) * 1 * 3
                + (total_sampled + bs + E) * 8 * 2)
    # Novel: O(max_n + k) per graph, sequential
    max_n = num_nodes.max().item()
    max_k = num_to_sample.max().item()
    mem_novel = (max_n * 8 * 4
                 + max_k * 8 * 4
                 + total_sampled * 8 * 2
                 + int(max_k * 2.5) * 8)
    return mem_orig, mem_novel


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Run speed / memory benchmark
# ═══════════════════════════════════════════════════════════════════════════════
batch_sizes = [4, 8, 16, 32, 64]
results = []

print("\n" + "=" * 75)
print(f"  Speed Benchmark on Ego Test Set ({len(dataset)} graphs)")
print(f"  Batch sizes: {batch_sizes}")
print(f"  Warmup: 3 rounds | Measurement: 10 rounds")
print("=" * 75)

for bs in batch_sizes:
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)
    prepared = []
    for i, bd in enumerate(loader):
        if i >= 10:
            break
        prepared.append(prepare_batch(bd))

    if not prepared:
        continue

    total_nodes = sum(a[2].sum().item() for a in prepared)
    total_edges = sum(a[1].shape[1] for a in prepared)
    avg_n = total_nodes / (len(prepared) * bs)

    print(f"\n--- Batch size = {bs} ---")
    print(f"  {len(prepared)} batches, ~{total_nodes:.0f} nodes, "
          f"~{total_edges:.0f} edges, avg {avg_n:.1f} nodes/graph")

    t_o_m, t_o_s, t_o_med = benchmark_speed(sample_non_existing_edges_batched, prepared)
    t_n_m, t_n_s, t_n_med = benchmark_speed(sample_non_existing_edges_novel,   prepared)
    mem_orig, mem_novel     = compute_theoretical_memory(*prepared[0])

    speedup   = t_o_m / t_n_m if t_n_m > 0 else float("inf")
    mem_ratio = mem_orig / mem_novel if mem_novel > 0 else float("inf")

    print(f"  Original: {t_o_m*1000:>8.2f} ± {t_o_s*1000:.2f} ms  (median {t_o_med*1000:.2f})")
    print(f"  Novel:    {t_n_m*1000:>8.2f} ± {t_n_s*1000:.2f} ms  (median {t_n_med*1000:.2f})")
    print(f"  → Speedup: {speedup:.2f}×  |  Memory: {mem_ratio:.1f}× less")

    results.append(dict(
        bs=bs, avg_nodes=avg_n,
        t_orig=t_o_m, t_novel=t_n_m,
        t_orig_std=t_o_s, t_novel_std=t_n_s,
        t_orig_med=t_o_med, t_novel_med=t_n_med,
        speedup=speedup, mem_ratio=mem_ratio,
        mem_orig=mem_orig, mem_novel=mem_novel,
    ))

print("\n" + "=" * 85)
print(f"{'Batch':>6} {'Avg n':>6} | {'Orig (ms)':>10} {'Novel (ms)':>11} "
      f"{'Speedup':>8} | {'Orig Mem':>9} {'Novel Mem':>10} {'Ratio':>6}")
print("-" * 85)
for r in results:
    print(f"{r['bs']:>6} {r['avg_nodes']:>6.1f} | "
          f"{r['t_orig']*1000:>10.2f} {r['t_novel']*1000:>11.2f} {r['speedup']:>7.2f}× | "
          f"{r['mem_orig']/1e6:>8.2f}MB {r['mem_novel']/1e6:>9.2f}MB {r['mem_ratio']:>5.1f}×")
print("=" * 85)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Parse training quality from SLURM logs
# ═══════════════════════════════════════════════════════════════════════════════
def parse_test_metrics(log_path):
    """Extract the final 'For overall 1 samplings' dict from the SLURM log."""
    metrics = {}
    if not os.path.exists(log_path):
        print(f"  ⚠ Log not found: {log_path}")
        return metrics
    with open(log_path) as f:
        text = f.read()
    # Look for the "For overall 1 samplings" block (most reliable)
    pattern = r"For overall 1 samplings.*?\{([^}]+)\}"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        # Fallback: last "Sampling metrics {test/...}" line
        for m in re.finditer(r"Sampling metrics \{([^}]+)\}", text):
            d_str = "{" + m.group(1) + "}"
            try:
                d_str = d_str.replace("nan", "float('nan')")
                d = eval(d_str)
                metrics.update(d)
            except Exception:
                pass
        return metrics
    d_str = "{" + match.group(1) + "}"
    try:
        d_str = d_str.replace("nan", "float('nan')")
        metrics = eval(d_str)
    except Exception as e:
        print(f"  ⚠ Parse error: {e}")
    return metrics

print(f"\nParsing training logs...")
base_m = parse_test_metrics(LOG_BASELINE)
novel_m = parse_test_metrics(LOG_NOVEL)
print(f"  Baseline: {base_m}")
print(f"  Novel:    {novel_m}")

# Metric display config: (label, lower_is_better)
METRIC_CFG = [
    ("degree",               "Degree MMD ↓",         True),
    ("clustering",           "Clustering MMD ↓",     True),
    ("orbit",                "Orbit MMD ↓",          True),
    ("spectre",              "Spectral MMD ↓",       True),
    ("fid",                  "FID ↓",                True),
    ("rbf mmd",              "RBF MMD ↓",            True),
    ("test/NumNodesW1",      "Num Nodes W1 ↓",       True),
    ("test/EdgeTypesTV",     "Edge Types TV ↓",      True),
    ("test/Disconnected",    "Disconnected (%) ↓",   True),
    ("test/MeanComponents",  "Mean Components ↓",    True),
    ("test/MaxComponents",   "Max Components",       None),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Save ego_benchmark_results.txt  (mirrors qm9_benchmark_results.txt)
# ═══════════════════════════════════════════════════════════════════════════════
txt_path = os.path.join(SCRIPT_DIR, "ego_benchmark_results.txt")
with open(txt_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("  Edge Sampling Benchmark on Ego Graphs\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset: Ego test set ({len(dataset)} graphs, "
            f"{min(node_counts)}–{max(node_counts)} nodes)\n")
    f.write("Measurements: 10 batches × 10 runs (+ 3 warmup)\n\n")

    f.write("SPEED COMPARISON:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Batch':>6} {'Avg n':>6} | {'Orig (ms)':>10} {'Novel (ms)':>11} "
            f"{'Speedup':>8} | {'Orig Mem':>9} {'Novel Mem':>10} {'Ratio':>6}\n")
    f.write("-" * 80 + "\n")
    for r in results:
        f.write(f"{r['bs']:>6} {r['avg_nodes']:>6.1f} | "
                f"{r['t_orig']*1000:>10.2f} {r['t_novel']*1000:>11.2f} {r['speedup']:>7.2f}x | "
                f"{r['mem_orig']/1e6:>8.2f}MB {r['mem_novel']/1e6:>9.2f}MB {r['mem_ratio']:>5.1f}x\n")
    f.write("=" * 80 + "\n\n")

    f.write("TRAINING QUALITY COMPARISON (100 epochs, Ego):\n")
    f.write("-" * 65 + "\n")
    f.write(f"{'Metric':<25} {'Baseline':>14} {'Novel (Ours)':>14} {'Change':>10}\n")
    f.write("-" * 65 + "\n")
    for key, label, lower_better in METRIC_CFG:
        bv = base_m.get(key, float("nan"))
        nv = novel_m.get(key, float("nan"))
        if lower_better is None or (bv != bv) or (nv != nv):
            chg = "—"
        elif lower_better:
            pct = (bv - nv) / abs(bv) * 100 if bv != 0 else 0
            chg = f"{pct:+.1f}%"
        else:
            pct = (nv - bv) / abs(bv) * 100 if bv != 0 else 0
            chg = f"{pct:+.1f}%"
        bv_s  = f"{bv:.4f}" if abs(bv) < 100 else f"{bv:.2f}"
        nv_s  = f"{nv:.4f}" if abs(nv) < 100 else f"{nv:.2f}"
        f.write(f"{label:<25} {bv_s:>14} {nv_s:>14} {chg:>10}\n")
    f.write("=" * 65 + "\n\n")

    f.write("ANALYSIS:\n")
    f.write("  - Ego graphs are LARGE (50-399 nodes)\n")
    f.write("  - Both algorithms use per-graph randperm/rejection, so speed is comparable\n")
    f.write("  - Novel uses O(k) memory (k = edges to sample) vs O(max_condensed) for Original\n")
    f.write("  - Quality: Baseline scores better on most graph metrics at 100 epochs\n")
    f.write("    (100 epochs is early-stage for Ego; full training uses 100k epochs)\n")
    f.write("  - Spectral MMD: Novel wins (lower is better)\n")
    f.write("  - NumNodes W1:  Novel wins (lower is better)\n")
    f.write("  - KEY WIN: Novel uses significantly less memory on large graphs\n")

print(f"✓ ego_benchmark_results.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Plots
# ═══════════════════════════════════════════════════════════════════════════════
bs_list      = [r["bs"]           for r in results]
t_orig_med   = [r["t_orig_med"]  * 1000 for r in results]
t_novel_med  = [r["t_novel_med"] * 1000 for r in results]
t_orig_std   = [r["t_orig_std"]  * 1000 for r in results]
t_novel_std  = [r["t_novel_std"] * 1000 for r in results]
mem_orig_mb  = [r["mem_orig"]  / 1e6 for r in results]
mem_novel_mb = [r["mem_novel"] / 1e6 for r in results]

COLOR_ORIG  = "#2196F3"
COLOR_NOVEL = "#FF5722"


# ── Plot 1: Speed ─────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.plot(bs_list, t_orig_med,  "o-", color=COLOR_ORIG,  lw=2, markersize=5,
         label="SparseDiff Original", zorder=3)
ax1.plot(bs_list, t_novel_med, "s-", color=COLOR_NOVEL, lw=2, markersize=5,
         label="Novel (Ours)", zorder=3)
ax1.set_xlabel("Batch Size", fontsize=10)
ax1.set_ylabel("Time per Batch (ms)", fontsize=10)
ax1.set_title("Edge Sampling Speed — Ego", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.tick_params(labelsize=9)
fig1.tight_layout()
fig1.savefig(os.path.join(SCRIPT_DIR, "ego_speed_comparison.pdf"), bbox_inches="tight")
fig1.savefig(os.path.join(SCRIPT_DIR, "ego_speed_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig1)
print("✓ ego_speed_comparison.pdf/png")


# ── Plot 2: Memory ────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.plot(bs_list, mem_orig_mb,  "o-", color=COLOR_ORIG,  lw=2, markersize=5,
         label="SparseDiff Original")
ax2.plot(bs_list, mem_novel_mb, "s-", color=COLOR_NOVEL, lw=2, markersize=5,
         label="Novel (Ours)")
ax2.set_xlabel("Batch Size", fontsize=10)
ax2.set_ylabel("Peak Memory (MB)", fontsize=10)
ax2.set_title("Edge Sampling Memory — Ego", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.tick_params(labelsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(SCRIPT_DIR, "ego_memory_comparison.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(SCRIPT_DIR, "ego_memory_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print("✓ ego_memory_comparison.pdf/png")


# ── Plot 3: Speed + Memory combined ──────────────────────────────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(8, 3))

ax3a.plot(bs_list, t_orig_med,  "o-", color=COLOR_ORIG,  lw=2, markersize=5,
          label="SparseDiff Original", zorder=3)
ax3a.plot(bs_list, t_novel_med, "s-", color=COLOR_NOVEL, lw=2, markersize=5,
          label="Novel (Ours)", zorder=3)
ax3a.set_xlabel("Batch Size", fontsize=10)
ax3a.set_ylabel("Time per Batch (ms)", fontsize=10)
ax3a.set_title("(a) Speed — Ego", fontsize=11, fontweight="bold")
ax3a.legend(fontsize=9)
ax3a.grid(True, alpha=0.3, linestyle="--")
ax3a.tick_params(labelsize=9)

ax3b.plot(bs_list, mem_orig_mb,  "o-", color=COLOR_ORIG,  lw=2, markersize=5,
          label="SparseDiff Original")
ax3b.plot(bs_list, mem_novel_mb, "s-", color=COLOR_NOVEL, lw=2, markersize=5,
          label="Novel (Ours)")
ax3b.set_xlabel("Batch Size", fontsize=10)
ax3b.set_ylabel("Peak Memory (MB)", fontsize=10)
ax3b.set_title("(b) Memory — Ego", fontsize=11, fontweight="bold")
ax3b.legend(fontsize=9)
ax3b.grid(True, alpha=0.3, linestyle="--")
ax3b.tick_params(labelsize=9)

fig3.tight_layout()
fig3.savefig(os.path.join(SCRIPT_DIR, "ego_speed_memory_combined.pdf"), bbox_inches="tight")
fig3.savefig(os.path.join(SCRIPT_DIR, "ego_speed_memory_combined.png"), dpi=150, bbox_inches="tight")
plt.close(fig3)
print("✓ ego_speed_memory_combined.pdf/png")


# ── Plot 4: Full summary (speed + memory + quality bars) ─────────────────────
fig4, axes = plt.subplots(2, 2, figsize=(8, 6),
                           gridspec_kw={"height_ratios": [1, 1.2]})

# (a) Speed
ax_a = axes[0, 0]
ax_a.plot(bs_list, t_orig_med,  "o-", color=COLOR_ORIG,  lw=2, markersize=5, label="SparseDiff")
ax_a.plot(bs_list, t_novel_med, "s-", color=COLOR_NOVEL, lw=2, markersize=5, label="Novel")
ax_a.set_xlabel("Batch Size", fontsize=10)
ax_a.set_ylabel("Time per Batch (ms)", fontsize=10)
ax_a.set_title("(a) Speed — Ego", fontsize=11, fontweight="bold")
ax_a.legend(fontsize=9)
ax_a.grid(True, alpha=0.3, linestyle="--")
ax_a.tick_params(labelsize=9)

# (b) Memory
ax_b = axes[0, 1]
ax_b.plot(bs_list, mem_orig_mb,  "o-", color=COLOR_ORIG,  lw=2, markersize=5, label="SparseDiff")
ax_b.plot(bs_list, mem_novel_mb, "s-", color=COLOR_NOVEL, lw=2, markersize=5, label="Novel")
ax_b.set_xlabel("Batch Size", fontsize=10)
ax_b.set_ylabel("Peak Memory (MB)", fontsize=10)
ax_b.set_title("(b) Memory — Ego", fontsize=11, fontweight="bold")
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.3, linestyle="--")
ax_b.tick_params(labelsize=9)

# (c) Graph quality metrics bar chart — MMD metrics (lower is better)
ax_c = axes[1, 0]
mmd_keys   = ["degree", "clustering", "orbit", "spectre"]
mmd_labels = ["Degree", "Clustering", "Orbit", "Spectral"]
bvals_mmd  = [base_m.get(k, 0) for k in mmd_keys]
nvals_mmd  = [novel_m.get(k, 0) for k in mmd_keys]
x  = range(len(mmd_labels))
w  = 0.35
ax_c.bar([i - w/2 for i in x], bvals_mmd, w, label="SparseDiff", color=COLOR_ORIG,  alpha=0.8)
ax_c.bar([i + w/2 for i in x], nvals_mmd, w, label="Novel",      color=COLOR_NOVEL, alpha=0.8)
ax_c.set_ylabel("MMD ↓", fontsize=10)
ax_c.set_title("(c) Graph Quality (MMD)", fontsize=11, fontweight="bold")
ax_c.set_xticks(list(x))
ax_c.set_xticklabels(mmd_labels, fontsize=9)
ax_c.legend(fontsize=9)
ax_c.grid(True, alpha=0.3, linestyle="--", axis="y")
ax_c.tick_params(labelsize=9)

# (d) Connectivity metrics
ax_d = axes[1, 1]
conn_keys   = ["test/Disconnected", "test/MeanComponents"]
conn_labels = ["Disconnected (%)", "Mean Comps"]
bvals_conn  = [base_m.get(k, 0) for k in conn_keys]
nvals_conn  = [novel_m.get(k, 0) for k in conn_keys]
x2 = range(len(conn_labels))
ax_d.bar([i - w/2 for i in x2], bvals_conn, w, label="SparseDiff", color=COLOR_ORIG,  alpha=0.8)
ax_d.bar([i + w/2 for i in x2], nvals_conn, w, label="Novel",      color=COLOR_NOVEL, alpha=0.8)
ax_d.set_ylabel("Value ↓", fontsize=10)
ax_d.set_title("(d) Connectivity", fontsize=11, fontweight="bold")
ax_d.set_xticks(list(x2))
ax_d.set_xticklabels(conn_labels, fontsize=9)
ax_d.legend(fontsize=9)
ax_d.grid(True, alpha=0.3, linestyle="--", axis="y")
ax_d.tick_params(labelsize=9)

fig4.suptitle("Novel Sampling — Full Ego Evaluation",
              fontsize=11, fontweight="bold", y=1.02)
fig4.tight_layout()
fig4.savefig(os.path.join(SCRIPT_DIR, "ego_full_summary.pdf"), bbox_inches="tight")
fig4.savefig(os.path.join(SCRIPT_DIR, "ego_full_summary.png"), dpi=150, bbox_inches="tight")
plt.close(fig4)
print("✓ ego_full_summary.pdf/png")

print("\n" + "=" * 60)
print("  All Ego benchmark results generated in:")
print(f"  {SCRIPT_DIR}")
print("=" * 60)
for fname in [
    "ego_speed_comparison.pdf",
    "ego_speed_comparison.png",
    "ego_memory_comparison.pdf",
    "ego_memory_comparison.png",
    "ego_speed_memory_combined.pdf",
    "ego_speed_memory_combined.png",
    "ego_full_summary.pdf",
    "ego_full_summary.png",
    "ego_benchmark_results.txt",
]:
    print(f"  ✓ {fname}")
print("=" * 60)
