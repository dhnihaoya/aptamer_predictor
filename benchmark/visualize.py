"""Visualize benchmark results as publication-quality line chart.

Usage:
  python benchmark/visualize.py                         # use mock data
  python benchmark/visualize.py -i benchmark/results.csv  # use real data
  python benchmark/visualize.py -i benchmark/results.csv -o benchmark/fig.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": False,
    "axes.grid": False,
})

MODE_CONFIG = {
    "original": {
        "label": "Original (no opt.)",
        "color": "#455a64",
        "marker": "o",
    },
    "vectorized": {
        "label": "Vectorized only",
        "color": "#1565c0",
        "marker": "s",
    },
    "cascade": {
        "label": "Cascade only",
        "color": "#2e7d32",
        "marker": "^",
    },
    "cuda": {
        "label": "CUDA only",
        "color": "#6a1b9a",
        "marker": "D",
    },
    "full": {
        "label": "Full acceleration",
        "color": "#e65100",
        "marker": "p",
    },
}

TIMEOUT_SEC = 3600


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Row:
    n_sites: int
    mode: str
    elapsed: Optional[float]  # None = timeout or >= threshold


def _clip(rows: list[Row], threshold: float) -> list[Row]:
    """Treat elapsed >= threshold as timeout."""
    out = []
    for r in rows:
        if r.elapsed is not None and r.elapsed >= threshold:
            out.append(Row(r.n_sites, r.mode, None))
        else:
            out.append(r)
    return out


def read_csv(path: str) -> list[Row]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            sec = r["elapsed_seconds"]
            rows.append(Row(
                n_sites=int(r["n_sites"]),
                mode=r["mode"],
                elapsed=None if sec == "TIMEOUT" else float(sec),
            ))
    return rows


def mock_data() -> list[Row]:
    """Generate realistic mock data for testing the plot."""
    site_counts = [7, 8, 9, 10, 11, 12, 13]
    base = {
        "original":   [8,   35,    145,    580,   2320,  None,  None],
        "vectorized": [0.3,  1.2,    5,      20,    80,    320,   1280],
        "cascade":    [1.2,  5,     20,      80,   320,   1280,   5120],
        "cuda":       [4,   16,     65,     260,  1040,   None,  None],
        "full":       [0.08, 0.3,    1.2,    4.8,   19,    76,    305],
    }
    rows = []
    for mode, times in base.items():
        for ns, t in zip(site_counts, times):
            rows.append(Row(ns, mode, t))
    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(
    rows: list[Row],
    output: Optional[str] = None,
    timeout: float = TIMEOUT_SEC,
    dpi: int = 300,
):
    rows = _clip(rows, timeout)
    site_counts = sorted(set(r.n_sites for r in rows))
    modes = [m for m in MODE_CONFIG if any(r.mode == m for r in rows)]

    fig, ax = plt.subplots(figsize=(7, 4.8))

    for mode in modes:
        cfg = MODE_CONFIG[mode]

        # Collect valid (non-timeout) points in site order
        valid_x, valid_y = [], []
        timeout_start = None

        for ns in site_counts:
            match = [r for r in rows if r.mode == mode and r.n_sites == ns]
            if not match:
                continue
            elapsed = match[0].elapsed
            if elapsed is not None:
                valid_x.append(ns)
                valid_y.append(elapsed)
            else:
                if timeout_start is None:
                    timeout_start = ns

        # Solid line for completed runs
        if valid_x:
            ax.plot(valid_x, valid_y,
                    color=cfg["color"], marker=cfg["marker"], markersize=7,
                    linewidth=1.8, label=cfg["label"], zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)

        # Dashed segment from last valid to timeout ceiling
        if timeout_start is not None and valid_x:
            to_x = [valid_x[-1], timeout_start]
            to_y = [valid_y[-1], timeout]
            ax.plot(to_x, to_y,
                    color=cfg["color"], linestyle="--",
                    linewidth=1.2, alpha=0.5, zorder=2)
            ax.scatter([timeout_start], [timeout],
                       color=cfg["color"], marker="^", s=50, zorder=4, alpha=0.7,
                       edgecolors="white", linewidths=0.8)

    # Timeout annotation
    ax.axhline(y=timeout, color="#999999", linestyle="--", linewidth=0.6, zorder=1)

    # Axes
    ax.set_xlabel("Number of mutation sites", fontsize=12)
    ax.set_ylabel("Elapsed time", fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(site_counts)
    ax.set_xticklabels([str(s) for s in site_counts], fontsize=11)

    # Nice time-scale ticks
    y_ticks = [0.1, 1, 10, 60, 600, 3600]
    y_labels = ["100ms", "1s", "10s", "1m", "10m", "1h"]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_ylim(0.05, 5000)
    ax.set_xlim(site_counts[0] - 0.5, site_counts[-1] + 0.5)

    # Subtle horizontal grid lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower right",
        fontsize=10,
        handlelength=2.5,
        handletextpad=0.4,
    )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def _fmt_y(val, _):
    if val >= 3600:
        return f"{val / 3600:.0f}h"
    if val >= 60:
        return f"{val / 60:.0f}m"
    if val >= 1:
        return f"{val:.0f}s"
    return f"{val * 1000:.0f}ms"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("-i", "--input", default=None,
                        help="CSV results file (omit to use mock data)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output image (fig.pdf / fig.png)")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC,
                        help="Timeout threshold in seconds (default: 3600)")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    if args.input:
        rows = read_csv(args.input)
    else:
        print("No input file — using mock data for preview")
        rows = mock_data()

    plot(rows, output=args.output, timeout=args.timeout, dpi=args.dpi)


if __name__ == "__main__":
    main()
