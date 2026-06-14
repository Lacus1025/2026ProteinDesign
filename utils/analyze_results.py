#!/usr/bin/env python3
"""
Analyze pipeline results across all experiments in res/.
Collects ALL sequences from ALL rounds globally, re-scores with 3 formulas
(no per-round normalization), and selects global top sequences with
evolution path tracing and distribution plots.
"""

import json
import os
import sys
import csv
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── output directory ──────────────────────────────────────────────
OUTPUT_DIR = "analysis_output"
TOP_N = 20


# ── data structures ───────────────────────────────────────────────


@dataclass
class EvolutionStep:
    round_num: int
    sequence: str
    parent: str | None
    brightness: float | None
    tm: float
    composite_score: float
    delta_tm: float


@dataclass
class GlobalEntry:
    sequence: str
    brightness: float
    tm: float
    delta_tm: float
    composite_score: float  # original pipeline score
    score_sq: float  # brightness² × ΔTm
    score_linear: float  # brightness × ΔTm
    score_bright: float  # brightness
    exp_name: str
    round_num: int
    parent: str | None = None


# ── loading ───────────────────────────────────────────────────────


def load_experiment(exp_dir: str) -> dict:
    json_path = os.path.join(exp_dir, "pipeline_results_sfGFP.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_experiments(res_dir: str = "res") -> list[str]:
    if not os.path.isdir(res_dir):
        raise FileNotFoundError(f"Directory not found: {res_dir}")
    return sorted(
        [
            os.path.join(res_dir, d)
            for d in os.listdir(res_dir)
            if os.path.isdir(os.path.join(res_dir, d))
        ]
    )


# ── lineage ───────────────────────────────────────────────────────


def build_lineage(data: dict) -> tuple[dict[str, dict], dict[str, dict], str, float]:
    lineage: dict[str, dict] = {}
    all_seqs: dict[str, dict] = {}

    for round_data in data["rounds"]:
        rnum = round_data["round"]
        parents = round_data["parents"]
        generated = round_data["generated"]

        idx = 0
        for parent_info in parents:
            parent_seq = parent_info["parent"]
            count = parent_info["generated_count"]
            for _ in range(count):
                if idx < len(generated):
                    child = generated[idx]
                    lineage[child["sequence"]] = {
                        "parent": parent_seq,
                        "round": rnum,
                        "entry": child,
                    }
                    all_seqs[child["sequence"]] = child
                idx += 1

    wt_seq = data["rounds"][0]["parents"][0]["parent"]
    wt_tm = data["summary"]["tm_wt"]
    return lineage, all_seqs, wt_seq, wt_tm


# ── global collection ─────────────────────────────────────────────


def collect_all_sequences(data: dict, exp_name: str) -> list[GlobalEntry]:
    """Extract every generated sequence from all rounds, annotated with experiment info."""
    entries: list[GlobalEntry] = []
    seq_seen: set[str] = set()

    for round_data in data["rounds"]:
        rnum = round_data["round"]
        parents = round_data["parents"]
        generated = round_data["generated"]

        idx = 0
        for parent_info in parents:
            parent_seq = parent_info["parent"]
            count = parent_info["generated_count"]
            for _ in range(count):
                if idx < len(generated):
                    g = generated[idx]
                    seq = g["sequence"]
                    if seq not in seq_seen:
                        seq_seen.add(seq)
                        entries.append(
                            GlobalEntry(
                                sequence=seq,
                                brightness=g["brightness"],
                                tm=g["tm"],
                                delta_tm=g["delta_tm"],
                                composite_score=g["composite_score"],
                                score_sq=0.0,
                                score_linear=0.0,
                                score_bright=g["brightness"],
                                exp_name=exp_name,
                                round_num=rnum,
                                parent=parent_seq,
                            )
                        )
                idx += 1

    return entries


def recompute_scores(entries: list[GlobalEntry]) -> None:
    """Re-score all entries with 3 non-normalized formulas."""
    for e in entries:
        b = e.brightness
        dtm = e.delta_tm
        e.score_sq = b * b * dtm
        e.score_linear = b * dtm
        e.score_bright = b


# ── global ranking ────────────────────────────────────────────────

SCORE_FIELDS = {
    "sq": ("score_sq", "Brightness² × ΔTm"),
    "linear": ("score_linear", "Brightness × ΔTm"),
    "bright": ("score_bright", "Brightness Only"),
}


def select_global_top(
    entries: list[GlobalEntry],
    score_key: str,
    top_n: int = TOP_N,
) -> list[GlobalEntry]:
    """Return top N entries sorted by a score field (descending)."""
    return sorted(entries, key=lambda x: getattr(x, score_key), reverse=True)[:top_n]


# ── evolution tracing ─────────────────────────────────────────────


def trace_evolution(
    target_seq: str,
    lineage: dict[str, dict],
    all_seqs: dict[str, dict],
    wt_seq: str,
    wt_tm: float,
) -> list[EvolutionStep]:
    steps: list[EvolutionStep] = []
    current = target_seq
    visited: set[str] = set()

    while current in lineage:
        if current in visited:
            print(
                f"  ⚠ Cycle detected in lineage for seq {current[:50]}... — breaking",
                file=sys.stderr,
            )
            break
        visited.add(current)
        info = lineage[current]
        entry = info["entry"]
        steps.append(
            EvolutionStep(
                round_num=info["round"],
                sequence=current,
                parent=info["parent"],
                brightness=entry["brightness"],
                tm=entry["tm"],
                composite_score=entry["composite_score"],
                delta_tm=entry["delta_tm"],
            )
        )
        current = info["parent"]

    steps.append(
        EvolutionStep(
            round_num=0,
            sequence=wt_seq,
            parent=None,
            brightness=None,
            tm=wt_tm,
            composite_score=0.0,
            delta_tm=0.0,
        )
    )
    steps.reverse()
    return steps


# ── diff formatting ───────────────────────────────────────────────


def format_diff_plain(parent: str | None, child: str) -> str:
    if parent is None:
        return f"{child[:30]}...  [WT]"
    marker_line: list[str] = []
    for pc, cc in zip(parent, child):
        marker_line.append("^" if pc != cc else " ")
    return "".join(marker_line)


# ── summary tables ────────────────────────────────────────────────


def print_global_top_table(
    top_entries: list[GlobalEntry],
    formula_label: str,
    score_field: str,
    file=None,
) -> None:
    header = (
        f"{'#':>3}  {'Experiment':<30} {'Rnd':>3}  {'Brightness':>10}  {'Tm (°C)':>8}  "
        f"{'ΔTm':>8}  {'Score':>12}  {'Seq (first 50)':<50}"
    )
    sep = "─" * len(header)
    lines = [
        f"\n{'=' * 120}",
        f"  GLOBAL TOP {len(top_entries)}  —  {formula_label}",
        f"{'=' * 120}",
        header,
        sep,
    ]

    for i, e in enumerate(top_entries, 1):
        short_name = e.exp_name.replace("res_", "").rstrip("/")
    for i, e in enumerate(top_entries, 1):
        short_name = e.exp_name.replace("res_", "").rstrip("/")
        score_val = getattr(e, score_field)
        lines.append(
            f"{i:>3}  {short_name:<30} {e.round_num:>3}  "
            f"{e.brightness:>10.4f}  {e.tm:>8.2f}  {e.delta_tm:>8.2f}  "
            f"{getattr(e, score_field):>12.4f}  {e.sequence[:50]:<50}"
        )

    output_text = "\n".join(lines)
    print(output_text)
    if file:
        file.write(output_text + "\n")


def print_evolution_path(
    exp_name: str,
    formula_label: str,
    rank: int,
    steps: list[EvolutionStep],
    file=None,
) -> None:
    short_name = exp_name.replace("res_", "").rstrip("/")
    lines = [
        f"\n{'─' * 100}",
        f"  Evolution Path #{rank}  |  {short_name}  |  {formula_label}",
        f"{'─' * 100}",
        f"{'Round':<7} {'Brightness':>12} {'Tm (°C)':>10} {'Composite':>11} {'ΔTm':>8}  Sequence (first 60, ^=mutation)",
        f"{'─' * 100}",
    ]

    for step in steps:
        bright_str = f"{step.brightness:.4f}" if step.brightness is not None else "N/A"
        rlabel = "WT" if step.round_num == 0 else str(step.round_num)
        seq_preview = step.sequence[:60]
        lines.append(
            f"R{rlabel:<6} {bright_str:>12} {step.tm:>10.2f} {step.composite_score:>11.4f} {step.delta_tm:>8.2f}  {seq_preview}"
        )
        if step.parent is not None:
            diff = format_diff_plain(step.parent, step.sequence)
            lines.append(f"{'':<56}  {diff[:60]}")

    output_text = "\n".join(lines)
    print(output_text)
    if file:
        file.write(output_text + "\n")


# ── CSV export ────────────────────────────────────────────────────


def export_global_csv(entries: list[GlobalEntry], output_dir: str) -> str:
    """Export all entries sorted by score_sq to a CSV file."""
    path = os.path.join(output_dir, "global_all_sequences.csv")
    sorted_entries = sorted(entries, key=lambda x: x.score_sq, reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "exp_name",
                "round",
                "brightness",
                "tm",
                "delta_tm",
                "score_sq",
                "score_linear",
                "score_bright",
                "composite_score_orig",
                "sequence",
            ]
        )
        for i, e in enumerate(sorted_entries, 1):
            writer.writerow(
                [
                    i,
                    e.exp_name,
                    e.round_num,
                    f"{e.brightness:.6f}",
                    f"{e.tm:.4f}",
                    f"{e.delta_tm:.4f}",
                    f"{e.score_sq:.6f}",
                    f"{e.score_linear:.6f}",
                    f"{e.score_bright:.6f}",
                    f"{e.composite_score:.6f}",
                    e.sequence,
                ]
            )
    return path


# ── distribution plots ────────────────────────────────────────────


def plot_distributions(entries: list[GlobalEntry], output_dir: str) -> list[str]:
    """Generate histogram and scatter distribution plots."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    brightness_vals = [e.brightness for e in entries]
    tm_vals = [e.tm for e in entries]
    delta_tm_vals = [e.delta_tm for e in entries]

    max_density = len(entries)  # for consistent y-axis limits across plots

    # determine bins using a unified FREEDMAN-DIACONIS-style approach
    def _hist_bins(data: list[float]) -> int:
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return 30
        bin_width = 2 * iqr / (len(data) ** (1 / 3))
        data_range = max(data) - min(data)
        if bin_width == 0:
            return 30
        return max(10, min(80, int(data_range / bin_width)))

    # Brightness histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        brightness_vals,
        bins=_hist_bins(brightness_vals),
        color="#2196F3",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        x=np.mean(brightness_vals),
        color="#1565C0",
        linestyle="--",
        linewidth=2,
        label=f"Mean={np.mean(brightness_vals):.3f}",
    )
    ax.set_xlabel("Predicted Brightness", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Global Brightness Distribution (All Experiments, All Rounds)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fpath = os.path.join(output_dir, "global_brightness_hist.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(fpath)

    # Tm histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        tm_vals,
        bins=_hist_bins(tm_vals),
        color="#FF5722",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        x=np.mean(tm_vals),
        color="#BF360C",
        linestyle="--",
        linewidth=2,
        label=f"Mean={np.mean(tm_vals):.1f}°C",
    )
    ax.set_xlabel("Predicted Tm (°C)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Global Tm Distribution (All Experiments, All Rounds)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fpath = os.path.join(output_dir, "global_tm_hist.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(fpath)

    # ΔTm histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        delta_tm_vals,
        bins=_hist_bins(delta_tm_vals),
        color="#4CAF50",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        x=np.mean(delta_tm_vals),
        color="#1B5E20",
        linestyle="--",
        linewidth=2,
        label=f"Mean={np.mean(delta_tm_vals):.1f}°C",
    )
    ax.set_xlabel("ΔTm (°C)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Global ΔTm Distribution (All Experiments, All Rounds)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fpath = os.path.join(output_dir, "global_delta_tm_hist.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(fpath)

    # Brightness vs ΔTm scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    # Sample for scatter if too many points
    sample_size = min(len(entries), 80000)
    if len(entries) > sample_size:
        indices = np.random.default_rng(42).choice(
            len(entries), sample_size, replace=False
        )
        b_sample = [brightness_vals[i] for i in indices]
        dtm_sample = [delta_tm_vals[i] for i in indices]
    else:
        b_sample = brightness_vals
        dtm_sample = delta_tm_vals

    scatter = ax.scatter(
        b_sample,
        dtm_sample,
        c=b_sample,
        cmap="viridis",
        alpha=0.15,
        s=2,
        edgecolors="none",
    )
    ax.set_xlabel("Predicted Brightness", fontsize=12)
    ax.set_ylabel("ΔTm (°C)", fontsize=12)
    ax.set_title(
        "Brightness vs ΔTm (All Experiments, All Rounds)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(y=0, color="grey", linestyle=":", alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Brightness", fontsize=10)
    ax.grid(True, alpha=0.2, linestyle=":")
    fig.tight_layout()
    fpath = os.path.join(output_dir, "global_brightness_vs_delta_tm.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(fpath)

    return paths


def plot_evolution_trends(
    exp_name: str,
    formula_key: str,
    rank: int,
    steps: list[EvolutionStep],
    output_dir: str,
) -> str:
    short_name = exp_name.replace("res_", "").rstrip("/")
    formula_label = SCORE_FIELDS[formula_key][1]

    rounds = [s.round_num for s in steps]
    brightness_vals = [
        s.brightness if s.brightness is not None else float("nan") for s in steps
    ]
    tm_vals = [s.tm for s in steps]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_b = "#2196F3"
    color_t = "#FF5722"

    ax1.set_xlabel("Evolution Round", fontsize=12)
    ax1.set_ylabel("Predicted Brightness", fontsize=12, color=color_b)
    ax1.plot(
        rounds,
        brightness_vals,
        marker="o",
        linestyle="-",
        linewidth=2,
        color=color_b,
        label="Brightness",
        zorder=3,
    )
    ax1.tick_params(axis="y", labelcolor=color_b)
    ax1.set_ylim(bottom=0)

    wt_brightness = steps[0].brightness
    if wt_brightness is not None:
        ax1.axhline(
            y=wt_brightness, color=color_b, linestyle="--", alpha=0.4, linewidth=1
        )
        ax1.text(
            max(rounds) * 0.98,
            wt_brightness,
            f" WT={wt_brightness:.2f}",
            color=color_b,
            fontsize=8,
            va="bottom",
            ha="right",
            alpha=0.7,
        )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Predicted Tm (°C)", fontsize=12, color=color_t)
    ax2.plot(
        rounds,
        tm_vals,
        marker="s",
        linestyle="-",
        linewidth=2,
        color=color_t,
        label="Tm",
        zorder=3,
    )
    ax2.tick_params(axis="y", labelcolor=color_t)

    wt_tm = steps[0].tm
    ax2.axhline(y=wt_tm, color=color_t, linestyle="--", alpha=0.4, linewidth=1)
    ax2.text(
        max(rounds) * 0.98,
        wt_tm,
        f" WT Tm={wt_tm:.1f}°C",
        color=color_t,
        fontsize=8,
        va="top",
        ha="right",
        alpha=0.7,
    )

    ax1.set_xticks(rounds)
    ax1.set_xticklabels(["WT" if r == 0 else str(r) for r in rounds])
    plt.title(
        f"Evolution Path #{rank} — {short_name} ({formula_label})",
        fontsize=14,
        fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{short_name}_{formula_key}_rank{rank:02d}_evolution.png"
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def plot_combined_summary(
    top_groups: dict[str, list[GlobalEntry]], output_dir: str
) -> str:
    """Bar chart: top-1 values per formula."""
    os.makedirs(output_dir, exist_ok=True)

    formulas = list(SCORE_FIELDS.keys())
    short_labels = [SCORE_FIELDS[k][1] for k in formulas]
    b_vals = [top_groups[k][0].brightness for k in formulas]
    tm_vals = [top_groups[k][0].tm for k in formulas]
    dtm_vals = [top_groups[k][0].delta_tm for k in formulas]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, vals, label, color in zip(
        axes,
        [b_vals, tm_vals, dtm_vals],
        ["Brightness", "Tm (°C)", "ΔTm (°C)"],
        ["#2196F3", "#FF5722", "#4CAF50"],
    ):
        bars = ax.bar(
            range(len(formulas)),
            vals,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.set_xticks(range(len(formulas)))
        ax.set_xticklabels(short_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"Best {label} per Formula", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(
        "Global #1 Sequence Comparison Across Scoring Formulas",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fpath = os.path.join(output_dir, "summary_comparison.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


# ── main ──────────────────────────────────────────────────────────


def main() -> None:
    res_dir = "res"  # 此处修改结果文件夹,默认输出为res文件夹下的所有实验结果.
    output_dir = OUTPUT_DIR
    top_n = TOP_N

    exp_dirs = discover_experiments(res_dir)

    if not exp_dirs:
        print("No experiment directories found in res/.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "analysis_report.txt")

    # ── Phase 1: Collect all sequences globally ─────────────────
    print(f"\n{'█' * 80}")
    print(f"  PHASE 1: Collecting ALL sequences from ALL experiments & rounds")
    print(f"{'█' * 80}")

    global_entries: list[GlobalEntry] = []
    exp_lineages: dict[str, tuple[dict, dict, str, float]] = {}

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"  Loading: {exp_name} ...", end=" ")
        try:
            data = load_experiment(exp_dir)
        except FileNotFoundError as exc:
            print(f"SKIP ({exc})")
            continue

        lineage, all_seqs, wt_seq, wt_tm = build_lineage(data)
        exp_lineages[exp_name] = (lineage, all_seqs, wt_seq, wt_tm)

        exp_entries = collect_all_sequences(data, exp_name)
        global_entries.extend(exp_entries)
        print(f"{len(exp_entries)} sequences")

    total = len(global_entries)
    print(f"\n  TOTAL sequences collected: {total}")

    # ── Phase 2: Re-score (no normalization) ────────────────────
    print(f"\n{'█' * 80}")
    print(f"  PHASE 2: Re-scoring with 3 non-normalized formulas")
    print(f"{'█' * 80}")
    recompute_scores(global_entries)
    print(f"  ✓ brightness² × ΔTm")
    print(f"  ✓ brightness × ΔTm")
    print(f"  ✓ brightness only")

    # ── Phase 3: Select global top N per formula ────────────────
    print(f"\n{'█' * 80}")
    print(f"  PHASE 3: Global Top {top_n} per scoring formula")
    print(f"{'█' * 80}")

    top_groups: dict[str, list[GlobalEntry]] = {}
    for fkey, (field, label) in SCORE_FIELDS.items():
        top = select_global_top(global_entries, field, top_n)
        top_groups[fkey] = top
        print(f"  {label}: top score = {getattr(top[0], field):.4f}")

    # ── Phase 4: Export CSV ─────────────────────────────────────
    print(f"\n{'█' * 80}")
    print(f"  PHASE 4: CSV Export")
    print(f"{'█' * 80}")
    csv_path = export_global_csv(global_entries, output_dir)
    print(f"  ✓ {csv_path}")

    # ── Phase 5: Generate report & evolution paths ──────────────
    print(f"\n{'█' * 80}")
    print(f"  PHASE 5: Generating report & evolution paths")
    print(f"{'█' * 80}")

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("GLOBAL SEQUENCE ANALYSIS REPORT\n")
        report.write("=" * 80 + "\n")
        report.write(f"Total sequences analyzed: {total}\n")
        report.write(
            f"Experiments: {', '.join(os.path.basename(d) for d in exp_dirs)}\n"
        )
        report.write(
            f"Scoring formulas: brightness²×ΔTm, brightness×ΔTm, brightness only\n"
        )
        report.write(f"Normalization: NONE (raw values only)\n")
        report.write("=" * 80 + "\n")

        for fkey, (field, label) in SCORE_FIELDS.items():
            top = top_groups[fkey]
            print_global_top_table(top, label, field, file=report)

            evo_dir = os.path.join(output_dir, "evolution", fkey)
            for rank, entry in enumerate(top, 1):
                exp_name = entry.exp_name
                if exp_name in exp_lineages:
                    lineage, all_seqs, wt_seq, wt_tm = exp_lineages[exp_name]
                    steps = trace_evolution(
                        entry.sequence, lineage, all_seqs, wt_seq, wt_tm
                    )
                    print_evolution_path(exp_name, label, rank, steps, file=report)
                    fpath = plot_evolution_trends(exp_name, fkey, rank, steps, evo_dir)
        report.write(f"\n\nCSV with full rankings: {csv_path}\n")

    print(f"  ✓ Report: {report_path}")

    # ── Phase 6: Distribution plots ─────────────────────────────
    print(f"\n{'█' * 80}")
    print(f"  PHASE 6: Distribution plots")
    print(f"{'█' * 80}")
    dist_paths = plot_distributions(global_entries, output_dir)
    for p in dist_paths:
        print(f"  ✓ {p}")

    # ── Phase 7: Summary comparison ─────────────────────────────
    print(f"\n{'█' * 80}")
    print(f"  PHASE 7: Summary comparison chart")
    print(f"{'█' * 80}")
    summary_path = plot_combined_summary(top_groups, output_dir)
    print(f"  ✓ {summary_path}")

    # ── Quick terminal summary ──────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  QUICK SUMMARY — Global #1 per formula")
    print(f"{'═' * 80}")
    for fkey, (field, label) in SCORE_FIELDS.items():
        e = top_groups[fkey][0]
        short_name = e.exp_name.replace("res_", "").rstrip("/")
        print(
            f"  {label:<26s}  B={e.brightness:.4f}  Tm={e.tm:.1f}°C  ΔTm={e.delta_tm:.1f}  "
            f"R{e.round_num}  {short_name}"
        )

    print(f"\nReport: {report_path}")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
