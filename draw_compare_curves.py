#!/usr/bin/env python3
"""
Plot TensorBoard scalar curves from one or more event files in a paper-like style.

This script supports two modes:
1) Built-in config mode (recommended for repeated experiments):
   edit the USER CONFIG block below, then run
       python3 plot_tfevents_curves.py
2) CLI mode:
   pass --tag / --group / --inputs / ... on the command line.

Typical built-in use
--------------------
Edit these fields in USER CONFIG:
- TAG
- GROUPS
- AGGREGATE / BAND / SMOOTH / TITLE / OUTPUT

Then simply run:
    python3 plot_tfevents_curves.py

Typical CLI use
---------------
1) List available scalar tags in an event file:
   python plot_tfevents_curves.py --list-tags /path/to/events.out.tfevents.xxx

2) Plot several runs of the same method together and aggregate by median:
   python plot_tfevents_curves.py \
       --tag "Info / Episode_Metric/Interception_Rate" \
       --group itad_v2 run1/events.out.tfevents.* run2/events.out.tfevents.* run3/events.out.tfevents.* \
       --group fixed fixed1/events.out.tfevents.* fixed2/events.out.tfevents.* \
       --aggregate median --band iqr --smooth 9 --output compare.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================= USER CONFIG =============================
# Set this to True if you want to run the script with no CLI arguments.
USE_BUILTIN_CONFIG = True

# Metric to plot
TAG = "Reward / Total reward (mean)"

# Group name -> list of event file paths / glob patterns
# Replace these example paths with your real paths.
GROUPS = {
    "Ours": [
        "/home/wcw/compare/checkpoints/2026-04-05_05-36-17_ippo_torch,3v3最新双教师完整训练，任务成功率95/events.out.tfevents.1775367379.wp.3822679.0",
    ],
    "Local only": [
        "/home/wcw/compare/checkpoints/2026-04-06_09-45-03_ippo_torch,3v3最新局部教师完整训练/events.out.tfevents.1775468707.wp.664371.0",
    ],
    "Global only": [
        "/home/wcw/compare/checkpoints/2026-04-07_13-51-48_ippo_torch,3v3最新全局教师完整训练/events.out.tfevents.1775569911.wp.1698263.0",
    ],
}

# Plot style / aggregation
RECURSIVE = True                 # allow ** globs
AGGREGATE = "median"            # median / mean / max
BAND = "iqr"                    # none / iqr / std
SMOOTH = 9                       # rolling window
INTERPOLATE = True               # align different runs onto a shared x-grid before aggregation
GRID_POINTS: Optional[int] = None

# Figure labels
XLABEL = "Timesteps"
YLABEL = "Total reward (mean)"
TITLE = "Comparison of Total reward (mean)"
X_IN_MILLIONS = True
LEGEND_LOC = "best"
FIGSIZE = (7.2, 4.8)
DPI = 180
OUTPUT = "/home/wcw/compare/outputs/Total_reward.png"
SHOW = False                     # True -> pop up window; False -> only save

# If you only want to inspect available tags in one file, put the file path here.
# Example: LIST_TAGS_FILE = "/mnt/data/events.out.tfevents.xxxxx"
LIST_TAGS_FILE: Optional[str] = None
# ======================================================================


def infer_event_files(paths: Sequence[str], recursive: bool = False) -> List[str]:
    files: List[str] = []
    for p in paths:
        matches = glob.glob(p, recursive=recursive)
        if not matches:
            if os.path.isfile(p):
                matches = [p]
            elif os.path.isdir(p):
                pattern = os.path.join(p, "**" if recursive else "", "events.out.tfevents.*")
                matches = glob.glob(pattern, recursive=recursive)
        for m in matches:
            if os.path.isdir(m):
                pattern = os.path.join(m, "**" if recursive else "", "events.out.tfevents.*")
                files.extend(glob.glob(pattern, recursive=recursive))
            elif os.path.basename(m).startswith("events.out.tfevents"):
                files.append(m)
    seen = set()
    ordered: List[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def method_name_from_path(path: str, level: int = 1) -> str:
    p = Path(path).resolve()
    parents = p.parents
    if level < len(parents):
        return parents[level - 1].name
    return p.parent.name or p.stem


def list_scalar_tags(event_file: str) -> List[str]:
    ea = EventAccumulator(event_file)
    ea.Reload()
    return list(ea.Tags().get("scalars", []))


def load_scalar(event_file: str, tag: str) -> pd.DataFrame:
    ea = EventAccumulator(event_file)
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    if tag not in scalar_tags:
        raise KeyError(f"Tag not found in {event_file}: {tag}")
    events = ea.Scalars(tag)
    return pd.DataFrame(
        {
            "step": [e.step for e in events],
            "value": [e.value for e in events],
            "wall_time": [e.wall_time for e in events],
        }
    )


def rolling_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return pd.Series(y).rolling(window=window, min_periods=1, center=False).mean().to_numpy()


def build_shared_grid(runs: List[pd.DataFrame], num_points: Optional[int] = None) -> np.ndarray:
    min_step = max(df["step"].min() for df in runs)
    max_step = min(df["step"].max() for df in runs)
    if min_step >= max_step:
        return np.unique(np.concatenate([df["step"].to_numpy() for df in runs]))
    if num_points is None:
        num_points = int(np.median([len(df) for df in runs]))
        num_points = max(num_points, 50)
    return np.linspace(min_step, max_step, num_points)


def interpolate_runs(runs: List[pd.DataFrame], grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for run_id, df in enumerate(runs):
        x = df["step"].to_numpy(dtype=float)
        y = df["value"].to_numpy(dtype=float)
        if len(x) == 0:
            continue
        if len(np.unique(x)) == 1:
            yi = np.full_like(grid, y[0], dtype=float)
        else:
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            yi = np.interp(grid, x, y)
        rows.append(pd.DataFrame({"step": grid, "value": yi, "run_id": run_id}))
    if not rows:
        return pd.DataFrame(columns=["step", "value", "run_id"])
    return pd.concat(rows, ignore_index=True)


def aggregate_runs(df: pd.DataFrame, aggregate: str = "median") -> pd.DataFrame:
    grouped = df.groupby("step")["value"]
    if aggregate == "median":
        center = grouped.median()
    elif aggregate == "mean":
        center = grouped.mean()
    elif aggregate == "max":
        center = grouped.max()
    else:
        raise ValueError(f"Unsupported aggregate: {aggregate}")

    out = pd.DataFrame({"step": center.index.to_numpy(), "center": center.to_numpy()})
    out["q25"] = grouped.quantile(0.25).reindex(center.index).to_numpy()
    out["q75"] = grouped.quantile(0.75).reindex(center.index).to_numpy()
    out["std"] = grouped.std(ddof=0).reindex(center.index).fillna(0.0).to_numpy()
    out["count"] = grouped.count().reindex(center.index).to_numpy()
    return out.sort_values("step").reset_index(drop=True)


def fmt_million(x: float, _pos=None) -> str:
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{x:.0f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TensorBoard scalar curves in a paper-like style")
    parser.add_argument("--tag", type=str, help="Scalar tag to plot, e.g. 'Info / Episode_Metric/Interception_Rate'")
    parser.add_argument("--inputs", nargs="*", default=[], help="Event files, directories, or glob patterns")
    parser.add_argument(
        "--group",
        nargs="+",
        action="append",
        metavar=("LABEL", "PATH"),
        help="Manually group runs under one method label. Example: --group itad_v2 run1/events.* run2/events.*",
    )
    parser.add_argument("--recursive", action="store_true", help="Enable recursive glob expansion for --inputs/--group paths")
    parser.add_argument("--aggregate", choices=["median", "mean", "max"], default="median")
    parser.add_argument("--band", choices=["none", "iqr", "std"], default="none")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling window for smoothing the plotted curves")
    parser.add_argument("--interpolate", action="store_true", help="Interpolate runs onto a shared step grid before aggregation")
    parser.add_argument("--grid-points", type=int, default=None, help="Number of shared grid points when --interpolate is enabled")
    parser.add_argument("--xlabel", type=str, default="Timesteps")
    parser.add_argument("--ylabel", type=str, default=None, help="Defaults to the tag name")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--output", type=str, default="curve.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--figsize", nargs=2, type=float, default=[7.2, 4.8], metavar=("W", "H"))
    parser.add_argument("--no-show", action="store_true", help="Do not show the plot interactively")
    parser.add_argument("--x-in-millions", action="store_true", help="Format x-axis as K/M")
    parser.add_argument("--legend-loc", type=str, default="best")
    parser.add_argument("--list-tags", type=str, default=None, help="Print scalar tags in one event file and exit")
    parser.add_argument("--label-level", type=int, default=1, help="Parent directory level used as auto label for --inputs")
    return parser.parse_args()


def build_config_from_builtin() -> argparse.Namespace:
    # mimic argparse.Namespace so the plotting pipeline can stay unified
    groups_as_cli = []
    for label, paths in GROUPS.items():
        groups_as_cli.append([label, *paths])

    return argparse.Namespace(
        tag=TAG,
        inputs=[],
        group=groups_as_cli,
        recursive=RECURSIVE,
        aggregate=AGGREGATE,
        band=BAND,
        smooth=SMOOTH,
        interpolate=INTERPOLATE,
        grid_points=GRID_POINTS,
        xlabel=XLABEL,
        ylabel=YLABEL,
        title=TITLE,
        output=OUTPUT,
        dpi=DPI,
        figsize=list(FIGSIZE),
        no_show=(not SHOW),
        x_in_millions=X_IN_MILLIONS,
        legend_loc=LEGEND_LOC,
        list_tags=LIST_TAGS_FILE,
        label_level=1,
    )


def run_plot(args: argparse.Namespace) -> None:
    if args.list_tags:
        tags = list_scalar_tags(args.list_tags)
        print("Scalar tags:")
        for t in tags:
            print(f" - {t}")
        return

    if not args.tag:
        raise SystemExit("Please provide --tag, or use --list-tags to inspect a file.")

    groups: Dict[str, List[str]] = {}

    if args.group:
        for group_spec in args.group:
            if len(group_spec) < 2:
                raise SystemExit("Each --group needs at least a LABEL and one PATH")
            label = group_spec[0]
            paths = group_spec[1:]
            files = infer_event_files(paths, recursive=args.recursive)
            if not files:
                print(f"[WARN] No event files found for group '{label}'")
                continue
            groups.setdefault(label, []).extend(files)

    if args.inputs:
        files = infer_event_files(args.inputs, recursive=args.recursive)
        for f in files:
            label = method_name_from_path(f, level=args.label_level)
            groups.setdefault(label, []).append(f)

    if not groups:
        raise SystemExit("No event files found. Check paths in USER CONFIG or CLI arguments.")

    method_curves: Dict[str, pd.DataFrame] = {}

    for label, files in groups.items():
        run_dfs: List[pd.DataFrame] = []
        for f in files:
            try:
                df = load_scalar(f, args.tag)
                if len(df) > 0:
                    run_dfs.append(df[["step", "value"]].copy())
            except Exception as e:
                print(f"[WARN] Skip {f}: {e}")

        if not run_dfs:
            print(f"[WARN] No valid runs for label '{label}' and tag '{args.tag}'")
            continue

        if args.interpolate and len(run_dfs) >= 2:
            grid = build_shared_grid(run_dfs, num_points=args.grid_points)
            merged = interpolate_runs(run_dfs, grid)
        else:
            frames = []
            for i, df in enumerate(run_dfs):
                tmp = df.copy()
                tmp["run_id"] = i
                frames.append(tmp)
            merged = pd.concat(frames, ignore_index=True)

        curve = aggregate_runs(merged, aggregate=args.aggregate)
        curve["center_s"] = rolling_smooth(curve["center"].to_numpy(), args.smooth)
        curve["q25_s"] = rolling_smooth(curve["q25"].to_numpy(), args.smooth)
        curve["q75_s"] = rolling_smooth(curve["q75"].to_numpy(), args.smooth)
        curve["std_s"] = rolling_smooth(curve["std"].to_numpy(), args.smooth)
        method_curves[label] = curve

    if not method_curves:
        raise SystemExit("No curves to plot after loading the requested tag.")

    plt.figure(figsize=tuple(args.figsize), dpi=args.dpi)

    for label in sorted(method_curves.keys()):
        curve = method_curves[label]
        x = curve["step"].to_numpy()
        y = curve["center_s"].to_numpy()
        line, = plt.plot(x, y, linewidth=2.2, label=label)

        if args.band == "iqr":
            plt.fill_between(x, curve["q25_s"].to_numpy(), curve["q75_s"].to_numpy(), alpha=0.16, color=line.get_color())
        elif args.band == "std":
            std = curve["std_s"].to_numpy()
            plt.fill_between(x, y - std, y + std, alpha=0.16, color=line.get_color())

    plt.xlabel(args.xlabel, fontsize=12)
    plt.ylabel(args.ylabel or args.tag, fontsize=12)
    if args.title:
        plt.title(args.title, fontsize=13)
    plt.legend(frameon=False, fontsize=9, loc=args.legend_loc)
    plt.grid(True, alpha=0.22)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if args.x_in_millions:
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_million))

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")

    if not args.no_show:
        plt.show()


def main() -> None:
    if USE_BUILTIN_CONFIG and len(sys.argv) == 1:
        args = build_config_from_builtin()
        run_plot(args)
    else:
        args = parse_args()
        run_plot(args)


if __name__ == "__main__":
    main()
