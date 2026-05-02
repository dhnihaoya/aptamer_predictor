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
        "color": "#4D4D4D",
        "marker": "o",
        "linestyle": "--",
        "hollow": True,
        "x_offset": -0.045,
        "zorder": 2,
    },
    "cuda": {
        "label": "CUDA only",
        "color": "#7B3294",
        "marker": "D",
        "linestyle": "-",
        "hollow": False,
        "x_offset": 0.045,
        "zorder": 3,
    },
    "vectorized": {
        "label": "Vectorization only",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
        "hollow": False,
        "x_offset": 0.0,
        "zorder": 4,
    },
    "cascade": {
        "label": "Cascade only",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
        "hollow": False,
        "x_offset": 0.0,
        "zorder": 4,
    },
    "full": {
        "label": "Full acceleration",
        "color": "#E69F00",
        "marker": "p",
        "linestyle": "-",
        "hollow": False,
        "x_offset": 0.0,
        "zorder": 5,
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

    all_valid_y = [r.elapsed for r in rows if r.elapsed is not None]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    for mode in modes:
        cfg = MODE_CONFIG[mode]
        x_off = cfg["x_offset"]

        # Collect valid (non-timeout) points in site order
        valid_x, valid_y = [], []
        timeout_start = None

        for ns in site_counts:
            match = [r for r in rows if r.mode == mode and r.n_sites == ns]
            if not match:
                continue
            elapsed = match[0].elapsed
            if elapsed is not None:
                valid_x.append(ns + x_off)
                valid_y.append(elapsed)
            else:
                if timeout_start is None:
                    timeout_start = ns

        # Line for completed runs
        if valid_x:
            mfc = "white" if cfg["hollow"] else cfg["color"]
            mec = cfg["color"] if cfg["hollow"] else "white"
            ax.plot(valid_x, valid_y,
                    color=cfg["color"], marker=cfg["marker"], markersize=7,
                    linewidth=1.8, linestyle=cfg["linestyle"],
                    label=cfg["label"], zorder=cfg["zorder"],
                    markerfacecolor=mfc, markeredgecolor=mec,
                    markeredgewidth=0.8)

        # Dashed segment from last valid to timeout ceiling
        if timeout_start is not None and valid_x:
            to_x = [valid_x[-1], timeout_start + x_off]
            to_y = [valid_y[-1], timeout]
            ax.plot(to_x, to_y,
                    color=cfg["color"], linestyle="--",
                    linewidth=1.2, alpha=0.5, zorder=2)
            ax.scatter([timeout_start + x_off], [timeout],
                       color=cfg["color"], marker="^", s=50, zorder=4, alpha=0.7,
                       edgecolors="white", linewidths=0.8)

    # Timeout annotation: subtle dotted line + text label
    ax.axhline(y=timeout, color="0.65", linestyle=":", linewidth=0.5, zorder=1)
    ax.text(site_counts[-1] + 0.3, timeout, "timeout",
            fontsize=8, color="0.55", va="center", ha="left")

    # Axes
    ax.set_xlabel("Number of mutation sites", fontsize=12)
    ax.set_ylabel("Elapsed time", fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(site_counts)
    ax.set_xticklabels([str(s) for s in site_counts], fontsize=11)

    # Y-axis: use _fmt_y formatter + minor ticks
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_y))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs=[2, 5], numticks=12)
    )

    # Adaptive y-limits
    if all_valid_y:
        y_lo = min(all_valid_y) / 3
        y_hi = max(max(all_valid_y), timeout) * 2
    else:
        y_lo, y_hi = 0.05, timeout * 2
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(site_counts[0] - 0.5, site_counts[-1] + 0.5)

    # Spines & grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, which="major", color="0.86", linewidth=0.5)
    ax.yaxis.grid(True, which="minor", color="0.93", linewidth=0.3)

    # Legend below the plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=10,
        handlelength=2.5,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

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
