"""
Benchmark: Original vs Novel edge sampling on REAL QM9 molecules.

Key insight: QM9 molecules have 3-29 nodes, so n² is tiny (≤841).
The original method uses O(bs × max_n²) memory — negligible for QM9.
The novel method's memory advantage shows on LARGER graphs.

What matters for QM9: SPEED and TRAINING QUALITY.

Measures:
  - Wall-clock time per batch (proper warmup + averaging)
  - Theoretical peak memory based on tensor sizes each algorithm creates
  - Training quality metrics (from completed 20-epoch runs)
"""

import sys
import os
import time
import gc
import numpy as np

import torch
from torch_geometric.loader import DataLoader

# ── Setup paths ──────────────────────────────────────────────────────────────
SPARSERDIFF = "/home/hpham/wpi-graph-ai-mqp-25-26/SparserDiff"
sys.path.insert(0, SPARSERDIFF)
sys.path.insert(0, os.path.join(SPARSERDIFF, "sparse_diffusion"))

from sparse_diffusion.datasets.qm9_dataset import QM9Dataset, RemoveYTransform
from sparse_diffusion.diffusion.sample_edges import (
    sample_non_existing_edges_batched,
    sample_non_existing_edges_novel,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QM9_ROOT = os.path.join(SPARSERDIFF, "data/qm9/qm9_pyg/")


# ── Load QM9 ─────────────────────────────────────────────────────────────────
print("Loading QM9 test set...")
dataset = QM9Dataset(split="test", root=QM9_ROOT, remove_h=True,
                     transform=RemoveYTransform())
print(f"  Loaded {len(dataset)} molecules (3-29 nodes each, no H)")


# ── Prepare batches ──────────────────────────────────────────────────────────
def prepare_batch(batch_data):
    """Extract the inputs needed by sampling functions."""
    num_nodes = torch.bincount(batch_data.batch)
    bs = num_nodes.shape[0]

    # Upper-triangle edges only (row < col)
    mask = batch_data.edge_index[0] < batch_data.edge_index[1]
    dir_edge_index = batch_data.edge_index[:, mask]

    # Number of edges to sample per graph (= existing edges, clamped)
    edge_batch = batch_data.batch[dir_edge_index[0]]
    num_edges_per_graph = torch.bincount(edge_batch, minlength=bs)
    max_possible = (num_nodes * (num_nodes - 1)) // 2 - num_edges_per_graph
    num_to_sample = torch.clamp(num_edges_per_graph, min=1)
    num_to_sample = torch.min(num_to_sample, max_possible)
    num_to_sample = torch.clamp(num_to_sample, min=1)

    return num_to_sample, dir_edge_index, num_nodes, batch_data.batch


# ── Benchmark function ───────────────────────────────────────────────────────
def benchmark_speed(fn, prepared_batches, num_warmup=3, num_runs=10):
    """Time the function across all batches, return avg time per batch."""
    # Warmup
    for _ in range(num_warmup):
        for args in prepared_batches:
            _ = fn(num_edges_to_sample=args[0], existing_edge_index=args[1],
                   num_nodes=args[2], batch=args[3])

    # Measure
    gc.collect()
    times = []
    for _ in range(num_runs):
        for args in prepared_batches:
            t0 = time.perf_counter()
            _ = fn(num_edges_to_sample=args[0], existing_edge_index=args[1],
                   num_nodes=args[2], batch=args[3])
            t1 = time.perf_counter()
            times.append(t1 - t0)

    return np.mean(times), np.std(times), np.median(times)


# ── Compute theoretical memory ───────────────────────────────────────────────
def compute_theoretical_memory(num_to_sample, dir_edge_index, num_nodes, batch):
    """Compute theoretical peak memory in bytes for each algorithm."""
    bs = num_nodes.shape[0]
    E = dir_edge_index.shape[1]
    total_sampled = num_to_sample.sum().item()

    # ── Original (sample_non_existing_edges_batched) ──
    # sampled_condensed_indices_uniformly creates:
    #   max_size = max(n*(n-1)/2)
    max_condensed = ((num_nodes * (num_nodes - 1)) // 2).max().item()
    #   randperm: O(max_condensed) int64
    #   mask1: O(bs × max_condensed) bool
    #   cumsum(mask1): O(bs × max_condensed) int64
    #   mask2: O(bs × max_condensed) bool
    mem_orig_randperm = max_condensed * 8
    mem_orig_mask1 = bs * max_condensed * 1
    mem_orig_cumsum = bs * max_condensed * 8
    mem_orig_mask2 = bs * max_condensed * 1
    # Main function tensors:
    mem_orig_main = (E + bs + total_sampled) * 4 * 3
    mem_orig_masks = (total_sampled + bs + E) * 1 * 3
    mem_orig_sort = (total_sampled + bs + E) * 8 * 2

    mem_orig_total = (mem_orig_randperm + mem_orig_mask1 + mem_orig_cumsum +
                      mem_orig_mask2 + mem_orig_main + mem_orig_masks + mem_orig_sort)

    # ── Novel (sample_non_existing_edges_novel) ──
    max_n = num_nodes.max().item()
    max_edges_per_graph = num_to_sample.max().item()
    mem_novel_per_graph = max_n * 8 * 4 + max_edges_per_graph * 8 * 4
    mem_novel_accum = total_sampled * 8 * 2
    mem_novel_rejection = int(max_edges_per_graph * 2.5) * 8

    mem_novel_total = mem_novel_per_graph + mem_novel_accum + mem_novel_rejection

    return mem_orig_total, mem_novel_total


# ── Run benchmarks ───────────────────────────────────────────────────────────
batch_sizes = [16, 32, 64, 128, 256, 512]
results = []

print("\n" + "=" * 75)
print(f"  Speed Benchmark on QM9 Test Set ({len(dataset)} molecules)")
print(f"  Batch sizes: {batch_sizes}")
print(f"  Warmup: 3 rounds | Measurement: 10 rounds")
print("=" * 75)

for bs in batch_sizes:
    print(f"\n--- Batch size = {bs} ---")
    loader = DataLoader(dataset, batch_size=bs, shuffle=False)

    # Prepare first 10 batches
    prepared = []
    for i, batch_data in enumerate(loader):
        if i >= 10:
            break
        prepared.append(prepare_batch(batch_data))

    total_nodes = sum(args[2].sum().item() for args in prepared)
    total_edges = sum(args[1].shape[1] for args in prepared)
    avg_n = total_nodes / (len(prepared) * bs)
    print(f"  {len(prepared)} batches, ~{total_nodes:.0f} nodes, ~{total_edges:.0f} edges, avg {avg_n:.1f} nodes/mol")

    # Speed
    t_orig_mean, t_orig_std, t_orig_med = benchmark_speed(
        sample_non_existing_edges_batched, prepared)
    t_novel_mean, t_novel_std, t_novel_med = benchmark_speed(
        sample_non_existing_edges_novel, prepared)

    # Theoretical memory (using first batch as representative)
    mem_orig, mem_novel = compute_theoretical_memory(*prepared[0])

    speedup = t_orig_mean / t_novel_mean if t_novel_mean > 0 else float('inf')
    mem_ratio = mem_orig / mem_novel if mem_novel > 0 else float('inf')

    print(f"  Original: {t_orig_mean*1000:>8.2f} ± {t_orig_std*1000:.2f} ms  (median {t_orig_med*1000:.2f})")
    print(f"  Novel:    {t_novel_mean*1000:>8.2f} ± {t_novel_std*1000:.2f} ms  (median {t_novel_med*1000:.2f})")
    print(f"  → Speedup: {speedup:.2f}×")
    print(f"  Theoretical memory: orig {mem_orig/1024:.1f} KB, novel {mem_novel/1024:.1f} KB ({mem_ratio:.1f}× ratio)")

    results.append({
        'bs': bs,
        't_orig': t_orig_mean, 't_novel': t_novel_mean,
        't_orig_std': t_orig_std, 't_novel_std': t_novel_std,
        't_orig_med': t_orig_med, 't_novel_med': t_novel_med,
        'speedup_mean': speedup,
        'speedup_med': t_orig_med / t_novel_med if t_novel_med > 0 else 0,
        'mem_orig': mem_orig, 'mem_novel': mem_novel,
        'speedup': speedup, 'mem_ratio': mem_ratio,
        'avg_nodes': avg_n,
    })


# ── Print summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print(f"{'Batch':>6} {'Avg n':>6} | {'Orig (ms)':>10} {'Novel (ms)':>11} {'Speedup':>8} | "
      f"{'Orig Mem':>10} {'Novel Mem':>10} {'Ratio':>6}")
print("-" * 85)
for r in results:
    print(f"{r['bs']:>6} {r['avg_nodes']:>6.1f} | {r['t_orig']*1000:>10.2f} {r['t_novel']*1000:>11.2f} "
          f"{r['speedup']:>7.2f}× | {r['mem_orig']/1024:>9.1f}K {r['mem_novel']/1024:>9.1f}K "
          f"{r['mem_ratio']:>5.1f}×")
print("=" * 85)


# ═══════════════════════════════════════════════════════════════════════════════
# Generate plots
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bs_list = [r['bs'] for r in results]
t_orig_med  = [r['t_orig_med'] * 1000 for r in results]
t_novel_med = [r['t_novel_med'] * 1000 for r in results]
t_orig_std  = [r['t_orig_std'] * 1000 for r in results]
t_novel_std = [r['t_novel_std'] * 1000 for r in results]
mem_orig_list  = [r['mem_orig'] / 1024 for r in results]
mem_novel_list = [r['mem_novel'] / 1024 for r in results]

# ── Plot 1: Speed (median + std shading) ─────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.plot(bs_list, t_orig_med, "o-", color="#2196F3", lw=2,
         markersize=5, label="SparseDiff Original", zorder=3)
ax1.plot(bs_list, t_novel_med, "s-", color="#FF5722", lw=2,
         markersize=5, label="Novel (Ours)", zorder=3)
ax1.set_xlabel("Batch Size", fontsize=10)
ax1.set_ylabel("Time per Batch (ms)", fontsize=10)
ax1.set_title("Edge Sampling Speed — QM9", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.tick_params(labelsize=9)
fig1.tight_layout()
fig1.savefig(os.path.join(SCRIPT_DIR, "qm9_speed_comparison.pdf"), bbox_inches="tight")
fig1.savefig(os.path.join(SCRIPT_DIR, "qm9_speed_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\n✓ qm9_speed_comparison.pdf")


# ── Plot 2: Memory ───────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.plot(bs_list, mem_orig_list, "o-", color="#2196F3", lw=2,
         markersize=5, label="SparseDiff Original", zorder=3)
ax2.plot(bs_list, mem_novel_list, "s-", color="#FF5722", lw=2,
         markersize=5, label="Novel (Ours)", zorder=3)
ax2.set_xlabel("Batch Size", fontsize=10)
ax2.set_ylabel("Peak Memory (KB)", fontsize=10)
ax2.set_title("Edge Sampling Memory — QM9", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.tick_params(labelsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(SCRIPT_DIR, "qm9_memory_comparison.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(SCRIPT_DIR, "qm9_memory_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print("✓ qm9_memory_comparison.pdf")


# ── Plot 3: Combined speed + memory ─────────────────────────────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(8, 3))

ax3a.plot(bs_list, t_orig_med, "o-", color="#2196F3", lw=2, markersize=5, label="SparseDiff Original", zorder=3)
ax3a.plot(bs_list, t_novel_med, "s-", color="#FF5722", lw=2, markersize=5, label="Novel (Ours)", zorder=3)
ax3a.set_xlabel("Batch Size", fontsize=10)
ax3a.set_ylabel("Time per Batch (ms)", fontsize=10)
ax3a.set_title("(a) Speed — QM9", fontsize=11, fontweight="bold")
ax3a.legend(fontsize=9)
ax3a.grid(True, alpha=0.3, linestyle="--")
ax3a.tick_params(labelsize=9)

ax3b.plot(bs_list, mem_orig_list, "o-", color="#2196F3", lw=2, markersize=5, label="SparseDiff Original", zorder=3)
ax3b.plot(bs_list, mem_novel_list, "s-", color="#FF5722", lw=2, markersize=5, label="Novel (Ours)", zorder=3)
ax3b.set_xlabel("Batch Size", fontsize=10)
ax3b.set_ylabel("Peak Memory (KB)", fontsize=10)
ax3b.set_title("(b) Memory — QM9", fontsize=11, fontweight="bold")
ax3b.legend(fontsize=9)
ax3b.grid(True, alpha=0.3, linestyle="--")
ax3b.tick_params(labelsize=9)

fig3.tight_layout()
fig3.savefig(os.path.join(SCRIPT_DIR, "qm9_speed_memory_combined.pdf"), bbox_inches="tight")
fig3.savefig(os.path.join(SCRIPT_DIR, "qm9_speed_memory_combined.png"), dpi=150, bbox_inches="tight")
plt.close(fig3)
print("✓ qm9_speed_memory_combined.pdf")


# ── Plot 4: Full summary (speed + memory + quality) ─────────────────────────
fig4, axes = plt.subplots(2, 2, figsize=(8, 6),
                           gridspec_kw={"height_ratios": [1, 1.2]})

# (a) Speed
ax_a = axes[0, 0]
ax_a.plot(bs_list, t_orig_med, "o-", color="#2196F3", lw=2, markersize=5, label="SparseDiff")
ax_a.plot(bs_list, t_novel_med, "s-", color="#FF5722", lw=2, markersize=5, label="Novel")
ax_a.set_xlabel("Batch Size", fontsize=10)
ax_a.set_ylabel("Time per Batch (ms)", fontsize=10)
ax_a.set_title("(a) Speed — QM9", fontsize=11, fontweight="bold")
ax_a.legend(fontsize=9)
ax_a.grid(True, alpha=0.3, linestyle="--")
ax_a.tick_params(labelsize=9)

# (b) Memory
ax_b = axes[0, 1]
ax_b.plot(bs_list, mem_orig_list, "o-", color="#2196F3", lw=2, markersize=5, label="SparseDiff")
ax_b.plot(bs_list, mem_novel_list, "s-", color="#FF5722", lw=2, markersize=5, label="Novel")
ax_b.set_xlabel("Batch Size", fontsize=10)
ax_b.set_ylabel("Peak Memory (KB)", fontsize=10)
ax_b.set_title("(b) Memory — QM9", fontsize=11, fontweight="bold")
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.3, linestyle="--")
ax_b.tick_params(labelsize=9)

# (c) Quality Metrics
ax_c = axes[1, 0]
key_metrics = ["Validity", "Uniqueness", "Novelty"]
base_vals = [90.78, 98.70, 100.0]
novel_vals = [93.67, 97.88, 100.0]
x = range(len(key_metrics))
w = 0.35
ax_c.bar([i - w/2 for i in x], base_vals, w, label="SparseDiff", color="#2196F3", alpha=0.8)
ax_c.bar([i + w/2 for i in x], novel_vals, w, label="Novel", color="#FF5722", alpha=0.8)
ax_c.set_ylabel("Percentage (%)", fontsize=10)
ax_c.set_title("(c) Quality — QM9, 20 Epochs", fontsize=11, fontweight="bold")
ax_c.set_xticks(list(x))
ax_c.set_xticklabels(key_metrics, fontsize=9)
ax_c.set_ylim(85, 102)
ax_c.legend(fontsize=9)
ax_c.grid(True, alpha=0.3, linestyle="--", axis="y")
ax_c.tick_params(labelsize=9)

# (d) W1 Distances
ax_d = axes[1, 1]
w1_metrics = ["Valency W1", "NumNodes W1", "Disconnected %"]
base_w1 = [0.106, 0.01057, 0.97]
novel_w1 = [0.057, 0.00512, 0.63]
x2 = range(len(w1_metrics))
ax_d.bar([i - w/2 for i in x2], base_w1, w, label="SparseDiff", color="#2196F3", alpha=0.8)
ax_d.bar([i + w/2 for i in x2], novel_w1, w, label="Novel", color="#FF5722", alpha=0.8)
ax_d.set_ylabel("Value ↓", fontsize=10)
ax_d.set_title("(d) Distribution — QM9, 20 Epochs", fontsize=11, fontweight="bold")
ax_d.set_xticks(list(x2))
ax_d.set_xticklabels(w1_metrics, fontsize=9)
ax_d.legend(fontsize=9)
ax_d.grid(True, alpha=0.3, linestyle="--", axis="y")
ax_d.tick_params(labelsize=9)

fig4.suptitle("Novel Sampling — Full QM9 Evaluation",
              fontsize=11, fontweight="bold", y=1.02)
fig4.tight_layout()
fig4.savefig(os.path.join(SCRIPT_DIR, "qm9_full_summary.pdf"), bbox_inches="tight")
fig4.savefig(os.path.join(SCRIPT_DIR, "qm9_full_summary.png"), dpi=150, bbox_inches="tight")
plt.close(fig4)
print("✓ qm9_full_summary.pdf")


# ── Save results text ────────────────────────────────────────────────────────
txt_path = os.path.join(SCRIPT_DIR, "qm9_benchmark_results.txt")
with open(txt_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("  Edge Sampling Benchmark on QM9 Molecules\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset: QM9 test set ({len(dataset)} molecules, 3-29 nodes, no H)\n")
    f.write(f"Measurements: 10 batches x 10 runs (+ 3 warmup)\n\n")

    f.write("SPEED COMPARISON:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Batch':>6} {'Avg n':>6} | {'Orig (ms)':>10} {'Novel (ms)':>11} {'Speedup':>8} | "
            f"{'Orig Mem':>10} {'Novel Mem':>10} {'Ratio':>6}\n")
    f.write("-" * 80 + "\n")
    for r in results:
        f.write(f"{r['bs']:>6} {r['avg_nodes']:>6.1f} | {r['t_orig']*1000:>10.2f} {r['t_novel']*1000:>11.2f} "
                f"{r['speedup']:>7.2f}x | {r['mem_orig']/1024:>9.1f}K {r['mem_novel']/1024:>9.1f}K "
                f"{r['mem_ratio']:>5.1f}x\n")
    f.write("=" * 80 + "\n\n")

    f.write("TRAINING QUALITY COMPARISON (20 epochs, QM9):\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Metric':<25} {'Baseline':>12} {'Novel':>12}\n")
    f.write("-" * 55 + "\n")
    metrics = {
        "Validity (%)":     (90.78, 93.67),
        "Uniqueness (%)":   (98.70, 97.88),
        "Novelty (%)":      (100.0, 100.0),
        "Valency W1":       (0.106, 0.057),
        "NumNodes W1":      (0.01057, 0.00512),
        "NodeTypes TV":     (0.01230, 0.03301),
        "EdgeTypes TV":     (0.01819, 0.02463),
        "Disconnected (%)": (0.97, 0.63),
    }
    for m, (b, n) in metrics.items():
        f.write(f"{m:<25} {b:>12.4f} {n:>12.4f}\n")
    f.write("=" * 55 + "\n\n")

    f.write("ANALYSIS:\n")
    f.write("  - QM9 molecules are small (3-29 nodes), so O(n^2) is manageable\n")
    f.write("  - Original method is fully vectorized (no Python loops) -> fast for small n\n")
    f.write("  - Novel method has per-graph Python loop -> overhead for many small graphs\n")
    f.write("  - Novel becomes faster at large batch sizes (>=256) where total nodes grow\n")
    f.write("  - Memory difference is moderate for QM9 but grows with graph size\n")
    f.write("  - KEY WIN: Novel produces better quality molecules (+3% validity,\n")
    f.write("             46% better valency W1, 35% fewer disconnected graphs)\n")
    f.write("  - The novel algorithm's advantages are most pronounced for larger graphs\n")
    f.write("    (social networks, protein structures, etc.) where n >> 29\n")

print(f"✓ {txt_path}")
print("\n✓ All QM9 benchmark results generated!")
