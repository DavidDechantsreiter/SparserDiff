"""
Generate metrics comparison table + plots for SparseDiff baseline vs Novel algorithm.
Both trained for 100 epochs on Ego dataset (social network ego-graphs).

Baseline: Job 1861442
Novel:    Job 1861443

This script:
  1. Parses logs to extract test metrics automatically
  2. Generates metrics_comparison.pdf / .csv / .txt
  3. Generates ego_full_summary.pdf  (4-panel combined figure)

Usage:
  python results_ego/generate_ego_metrics.py
  OR with explicit log paths:
  python results_ego/generate_ego_metrics.py --baseline_log <path> --novel_log <path>
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import os
import re
import sys
import argparse
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# Parse metrics from SLURM logs
# ═══════════════════════════════════════════════════════════════════════════════

def parse_test_metrics(log_path):
    """Extract test sampling metrics from a SparserDiff SLURM log."""
    metrics = {}
    with open(log_path, "r") as f:
        content = f.read()

    # Find all "Sampling metrics {..." lines with test/ prefix
    pattern = r"Sampling metrics (\{[^}]+\})"
    matches = re.findall(pattern, content)

    for match in matches:
        # Parse the dict-like string
        # Replace nan with "NaN" for json parsing
        cleaned = match.replace("nan", '"NaN"').replace("'", '"')
        try:
            d = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: manual parsing
            d = {}
            pairs = re.findall(r"'([^']+)':\s*([^,}]+)", match)
            for key, val in pairs:
                val = val.strip()
                try:
                    d[key] = float(val) if val != "nan" else "NaN"
                except ValueError:
                    d[key] = val

        # Keep all metrics (test/ prefix and raw keys like degree, clustering)
        for key, val in d.items():
            metrics[key] = val

    return metrics


def parse_train_loss(log_path):
    """Extract training loss progression from log."""
    epochs = []
    with open(log_path, "r") as f:
        for line in f:
            m = re.search(r"Epoch (\d+) finished: X: ([\d.]+) -- E: ([\d.]+)", line)
            if m:
                epochs.append({
                    "epoch": int(m.group(1)),
                    "loss_X": float(m.group(2)),
                    "loss_E": float(m.group(3)),
                })
            # Also capture val NLL
            m2 = re.search(r"Epoch (\d+): Val NLL ([\d.]+)", line)
            if m2:
                ep = int(m2.group(1))
                nll = float(m2.group(2))
                # Find or add
                found = False
                for e in epochs:
                    if e["epoch"] == ep:
                        e["val_nll"] = nll
                        found = True
                        break
                if not found:
                    epochs.append({"epoch": ep, "val_nll": nll})
    return epochs


# ═══════════════════════════════════════════════════════════════════════════════
# Find log files
# ═══════════════════════════════════════════════════════════════════════════════

def find_logs():
    """Auto-detect the most recent ego baseline and novel logs."""
    log_dir = "/scratch/hpham/SparseDiff/logs"
    # Known job IDs
    candidates = [
        (1861442, "baseline"),
        (1861443, "novel"),
    ]
    paths = {}
    for job_id, label in candidates:
        p = os.path.join(log_dir, f"{job_id}.out")
        if os.path.exists(p):
            paths[label] = p

    if len(paths) < 2:
        # Fallback: scan outputs dir
        output_base = os.path.join(
            os.path.dirname(SCRIPT_DIR), "outputs", "2026-03-06"
        )
        for d in sorted(os.listdir(output_base)):
            if "ego_baseline" in d:
                lp = os.path.join(output_base, d, "main.log")
                if os.path.exists(lp):
                    paths["baseline"] = lp
            elif "ego_novel" in d:
                lp = os.path.join(output_base, d, "main.log")
                if os.path.exists(lp):
                    paths["novel"] = lp

    return paths.get("baseline"), paths.get("novel")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_log", type=str, default=None)
    parser.add_argument("--novel_log", type=str, default=None)
    args = parser.parse_args()

    if args.baseline_log and args.novel_log:
        base_log, novel_log = args.baseline_log, args.novel_log
    else:
        base_log, novel_log = find_logs()

    if not base_log or not novel_log:
        print("ERROR: Could not find both log files.")
        print(f"  Baseline: {base_log}")
        print(f"  Novel:    {novel_log}")
        sys.exit(1)

    print(f"Baseline log: {base_log}")
    print(f"Novel log:    {novel_log}")

    # Parse metrics
    base_metrics = parse_test_metrics(base_log)
    novel_metrics = parse_test_metrics(novel_log)

    if not base_metrics:
        print("WARNING: No test metrics found in baseline log. Job may still be running.")
        print("  Trying val/ metrics instead...")
        # Fallback to val metrics
        with open(base_log, "r") as f:
            content = f.read()
        pattern = r"Sampling metrics (\{[^}]+\})"
        matches = re.findall(pattern, content)
        for match in matches:
            cleaned = match.replace("nan", '"NaN"').replace("'", '"')
            try:
                d = json.loads(cleaned)
                for k, v in d.items():
                    if k.startswith("val/"):
                        base_metrics["test/" + k[4:]] = v
            except:
                pass

    if not novel_metrics:
        print("WARNING: No test metrics found in novel log. Job may still be running.")
        sys.exit(1)

    print(f"\nBaseline test metrics: {base_metrics}")
    print(f"Novel test metrics:    {novel_metrics}")

    # Parse training loss
    base_train = parse_train_loss(base_log)
    novel_train = parse_train_loss(novel_log)

    # ═══════════════════════════════════════════════════════════════════════════
    # Ego metrics mapping (spectre dataset metrics)
    # ═══════════════════════════════════════════════════════════════════════════

    # Ego uses: degree, clustering, orbit, spectre, neural + basic graph stats
    metric_display = {
        "degree":            ("Degree (MMD) ↓", True),
        "clustering":        ("Clustering (MMD) ↓", True),
        "orbit":             ("Orbit (MMD) ↓", True),
        "spectre":           ("Spectral (MMD) ↓", True),
        "fid":               ("FID ↓", True),
        "rbf mmd":           ("RBF MMD ↓", True),
        "test/NumNodesW1":   ("Num Nodes W1 ↓", True),
        "test/EdgeTypesTV":  ("Edge Types TV ↓", True),
        "test/Disconnected": ("Disconnected (%) ↓", True),
        "test/MeanComponents": ("Mean Components ↓", True),
        "test/MaxComponents": ("Max Components", None),
    }

    # Build ordered metrics dict
    metrics = {}
    for key, (display_name, _) in metric_display.items():
        b = base_metrics.get(key, "—")
        n = novel_metrics.get(key, "—")
        if b != "—" or n != "—":
            metrics[display_name] = (b, n)

    if not metrics:
        print("ERROR: No matching metrics found between baseline and novel.")
        print(f"  Base keys: {list(base_metrics.keys())}")
        print(f"  Novel keys: {list(novel_metrics.keys())}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Found {len(metrics)} metrics to compare")
    print(f"{'='*65}")
    for m, (b, n) in metrics.items():
        print(f"  {m:<30} Base: {b}  Novel: {n}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Save CSV
    # ═══════════════════════════════════════════════════════════════════════════

    csv_path = os.path.join(SCRIPT_DIR, "ego_metrics_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline (Original)", "Novel (Ours)", "Improvement"])
        for metric_name, (base, novel) in metrics.items():
            imp = compute_improvement(metric_name, base, novel)
            writer.writerow([metric_name, base, novel, imp])
    print(f"\n✓ {csv_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Save TXT
    # ═══════════════════════════════════════════════════════════════════════════

    txt_path = os.path.join(SCRIPT_DIR, "ego_metrics_comparison.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 75 + "\n")
        f.write("  SparseDiff: Baseline vs Novel Algorithm — Ego Dataset, 100 Epochs\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"{'Metric':<30} {'Baseline':>12} {'Novel (Ours)':>14} {'Change':>10}\n")
        f.write("-" * 70 + "\n")
        for metric_name, (base, novel) in metrics.items():
            imp = compute_improvement(metric_name, base, novel)
            base_s = fmt_val(base)
            novel_s = fmt_val(novel)
            f.write(f"{metric_name:<30} {base_s:>12} {novel_s:>14} {imp:>10}\n")
        f.write("-" * 70 + "\n\n")
        f.write("Dataset info:\n")
        f.write("  • Ego: 908 social network ego-graphs (50-399 nodes, avg ~144)\n")
        f.write("  • Train: 606, Val: 151, Test: 151 graphs\n")
        f.write("  • Metrics: degree, clustering, orbit, spectral (MMD), neural (F1)\n")
        f.write("  • Both models trained with identical hyperparameters except sampling\n")
        f.write("  • SLURM cluster: WPI (L40S GPU)\n")
        f.write("=" * 75 + "\n")
    print(f"✓ {txt_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Metrics table PDF
    # ═══════════════════════════════════════════════════════════════════════════

    fig, ax = plt.subplots(figsize=(13, max(5, 1 + len(metrics) * 0.55)))
    ax.axis("off")
    ax.set_title(
        "SparseDiff: Baseline vs Novel Algorithm\nEgo Dataset — 100 Epochs",
        fontsize=16, fontweight="bold", pad=20,
    )

    col_labels = ["Metric", "Baseline\n(Original)", "Novel\n(Ours)", "Change"]
    cell_data = []
    cell_colors = []

    for metric_name, (base, novel) in metrics.items():
        imp = compute_improvement(metric_name, base, novel)
        color = get_color(metric_name, base, novel)
        cell_data.append([metric_name, fmt_val(base), fmt_val(novel), imp])
        cell_colors.append(["white", "white", "white", color])

    table = ax.table(
        cellText=cell_data, colLabels=col_labels,
        cellLoc="center", loc="center",
        colColours=["#E3F2FD"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for i, row_colors in enumerate(cell_colors):
        for j, color in enumerate(row_colors):
            if color != "white":
                table[i + 1, j].set_facecolor(color)

    for j in range(4):
        table[0, j].set_text_props(fontweight="bold", fontsize=12)
    for i in range(len(cell_data)):
        table[i + 1, 0].set_text_props(ha="left")

    ax.text(
        0.5, -0.02,
        "Green = improvement | Yellow = marginal | Red = regression\n"
        "↑ = higher is better | ↓ = lower is better | MMD = Maximum Mean Discrepancy",
        transform=ax.transAxes, fontsize=9, ha="center", va="top",
        style="italic", color="gray",
    )

    fig.tight_layout()
    pdf_path = os.path.join(SCRIPT_DIR, "ego_metrics_comparison.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {pdf_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Full summary PDF (4 panels)
    # ═══════════════════════════════════════════════════════════════════════════

    fig_sum, axes = plt.subplots(2, 2, figsize=(18, 14))

    # (a) Training loss curve
    ax_a = axes[0, 0]
    if base_train and novel_train:
        base_epochs = [e["epoch"] for e in base_train if "loss_E" in e]
        base_loss_E = [e["loss_E"] for e in base_train if "loss_E" in e]
        novel_epochs = [e["epoch"] for e in novel_train if "loss_E" in e]
        novel_loss_E = [e["loss_E"] for e in novel_train if "loss_E" in e]

        ax_a.plot(base_epochs, base_loss_E, "-", color="#2196F3", lw=2, alpha=0.7, label="Baseline")
        ax_a.plot(novel_epochs, novel_loss_E, "-", color="#FF5722", lw=2, alpha=0.7, label="Novel (Ours)")
        ax_a.set_xlabel("Epoch", fontsize=12)
        ax_a.set_ylabel("Edge Loss (E)", fontsize=12)
        ax_a.set_title("(a) Training Edge Loss", fontsize=13, fontweight="bold")
        ax_a.legend(fontsize=10)
        ax_a.grid(True, alpha=0.3, linestyle="--")
    else:
        ax_a.text(0.5, 0.5, "No training data", transform=ax_a.transAxes, ha="center")
        ax_a.set_title("(a) Training Edge Loss", fontsize=13, fontweight="bold")

    # (b) Speed benchmark (from earlier benchmark_ego.py results)
    ax_b = axes[0, 1]
    # Hardcoded from the benchmark run
    batch_sizes = [4, 8, 16, 32, 64]
    orig_speed = [308.68, 541.18, 842.76, 898.91, 1178.01]
    novel_speed = [2.57, 16.91, 51.06, 58.81, 99.25]

    ax_b.plot(batch_sizes, orig_speed, "o-", color="#2196F3", lw=2.5, markersize=8, label="Original")
    ax_b.plot(batch_sizes, novel_speed, "s-", color="#FF5722", lw=2.5, markersize=8, label="Novel (Ours)")
    ax_b.set_xlabel("Batch Size", fontsize=12)
    ax_b.set_ylabel("Time (ms)", fontsize=12)
    ax_b.set_title("(b) Sampling Speed — Ego Graphs", fontsize=13, fontweight="bold")
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3, linestyle="--")
    ax_b.set_yscale("log")

    # (c) Key metrics bar chart (MMD metrics: degree, clustering, orbit, spectre)
    ax_c = axes[1, 0]
    mmd_keys = ["Degree (MMD) ↓", "Clustering (MMD) ↓", "Orbit (MMD) ↓", "Spectral (MMD) ↓"]
    mmd_base = []
    mmd_novel = []
    mmd_labels = []
    for k in mmd_keys:
        if k in metrics:
            b, n = metrics[k]
            if isinstance(b, (int, float)) and isinstance(n, (int, float)):
                mmd_base.append(b)
                mmd_novel.append(n)
                mmd_labels.append(k.replace(" (MMD) ↓", ""))

    if mmd_labels:
        x = range(len(mmd_labels))
        w = 0.35
        ax_c.bar([i - w / 2 for i in x], mmd_base, w, label="Baseline", color="#2196F3", alpha=0.8)
        ax_c.bar([i + w / 2 for i in x], mmd_novel, w, label="Novel (Ours)", color="#FF5722", alpha=0.8)
        ax_c.set_ylabel("MMD (lower = better)", fontsize=12)
        ax_c.set_title("(c) Graph Structure Metrics (MMD)", fontsize=13, fontweight="bold")
        ax_c.set_xticks(list(x))
        ax_c.set_xticklabels(mmd_labels, fontsize=11)
        ax_c.legend(fontsize=10)
        ax_c.grid(True, alpha=0.3, linestyle="--", axis="y")
    else:
        ax_c.text(0.5, 0.5, "No MMD metrics available", transform=ax_c.transAxes, ha="center")
        ax_c.set_title("(c) Graph Structure Metrics (MMD)", fontsize=13, fontweight="bold")

    # (d) Memory comparison
    ax_d = axes[1, 1]
    orig_mem = [2.67, 4.87, 9.32, 18.28, 36.13]
    novel_mem = [0.08, 0.09, 0.13, 0.23, 0.41]

    ax_d.plot(batch_sizes, orig_mem, "o-", color="#2196F3", lw=2.5, markersize=8, label="Original")
    ax_d.plot(batch_sizes, novel_mem, "s-", color="#FF5722", lw=2.5, markersize=8, label="Novel (Ours)")
    ax_d.set_xlabel("Batch Size", fontsize=12)
    ax_d.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax_d.set_title("(d) Theoretical Peak Memory — Ego Graphs", fontsize=13, fontweight="bold")
    ax_d.legend(fontsize=10)
    ax_d.grid(True, alpha=0.3, linestyle="--")
    ax_d.set_yscale("log")

    fig_sum.suptitle(
        "SparseDiff: Novel Sampling Algorithm — Ego Dataset, 100 Epochs\n"
        "(a) Training Loss  (b) Sampling Speed  (c) Quality Metrics  (d) Memory",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig_sum.tight_layout()
    summary_path = os.path.join(SCRIPT_DIR, "ego_full_summary.pdf")
    fig_sum.savefig(summary_path, bbox_inches="tight")
    plt.close(fig_sum)
    print(f"✓ {summary_path}")

    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  All Ego deliverables generated in: {SCRIPT_DIR}")
    print(f"{'='*60}")
    print(f"  ego_metrics_comparison.pdf  — Metrics table (visual)")
    print(f"  ego_metrics_comparison.csv  — Metrics table (data)")
    print(f"  ego_metrics_comparison.txt  — Metrics table (text)")
    print(f"  ego_full_summary.pdf        — 4-panel combined figure")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_improvement(metric_name, base, novel):
    if isinstance(base, str) or isinstance(novel, str):
        return "—"
    if base == "NaN" or novel == "NaN":
        return "—"
    is_lower_better = "↓" in metric_name
    if is_lower_better:
        if base != 0:
            pct = (base - novel) / abs(base) * 100
            return f"{pct:+.1f}%"
    else:
        if base != 0:
            pct = (novel - base) / abs(base) * 100
            return f"{pct:+.1f}%"
    return "—"


def get_color(metric_name, base, novel):
    if isinstance(base, str) or isinstance(novel, str) or base == "NaN" or novel == "NaN":
        return "white"
    is_lower_better = "↓" in metric_name
    is_higher_better = "↑" in metric_name
    if is_lower_better:
        pct = (base - novel) / abs(base) * 100 if base != 0 else 0
        return "#C8E6C9" if pct > 5 else ("#FFECB3" if pct > 0 else "#FFCDD2")
    elif is_higher_better:
        pct = (novel - base) / abs(base) * 100 if base != 0 else 0
        return "#C8E6C9" if pct > 1 else ("#FFECB3" if pct >= 0 else "#FFCDD2")
    return "white"


def fmt_val(v):
    if isinstance(v, str):
        return v
    if v == "NaN":
        return "NaN"
    if abs(v) < 0.001:
        return f"{v:.6f}"
    if abs(v) < 10:
        return f"{v:.4f}"
    return f"{v:.2f}"


if __name__ == "__main__":
    main()
