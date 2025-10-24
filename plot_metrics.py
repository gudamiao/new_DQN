#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training metrics from multiple experiments for comparison.

Usage:
  # Plot results from specific directories
  python plot_metrics.py outputs/sc_base outputs/sc_full_model

  # Plot all scenarios found in the outputs directory
  python plot_metrics.py outputs/*
"""
import argparse
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob


def read_meta_if_any(csv_path):
    meta_path = os.path.join(os.path.dirname(csv_path), "run_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())


def get_col(df: pd.DataFrame, candidates):
    want = [_norm(x) for x in candidates]
    mapping = {_norm(c): c for c in df.columns}
    for w in want:
        if w in mapping:
            return mapping[w]
    return None


def plot_comparison(datasets, col_candidates, title, ylabel, window_size=50):
    plt.figure(figsize=(12, 8))

    for name, df in datasets.items():
        col_to_plot = get_col(df, col_candidates)
        if col_to_plot is not None:
            # 使用滑动平均使曲线更平滑
            y_smooth = df[col_to_plot].rolling(window=window_size, min_periods=1).mean()
            x_col = get_col(df, ["episode", "ep"])
            x = df[x_col] if x_col else range(1, len(df) + 1)
            plt.plot(x, y_smooth, label=name)
        else:
            print(f"[WARN] Skipping {name} for plot '{title}': Column not found from {col_candidates}")

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()


def main():
    ap = argparse.ArgumentParser(description="Plot and compare metrics from multiple DQN training runs.")
    ap.add_argument("paths", nargs='+',
                    help="List of paths to scenario directories (e.g., outputs/sc_base outputs/sc_full_model). Supports wildcards like outputs/*.")
    args = ap.parse_args()

    datasets = {}
    # Expand wildcards and find CSVs
    for path_pattern in args.paths:
        for path in glob.glob(path_pattern):
            if not os.path.isdir(path):
                continue

            scenario_name = os.path.basename(path).replace('sc_', '')
            csv_path = os.path.join(path, "metrics_multi_roles.csv")

            if os.path.isfile(csv_path):
                print(f"Loading data for '{scenario_name}' from {csv_path}")
                df = pd.read_csv(csv_path)
                datasets[scenario_name] = df
            else:
                print(f"[WARN] No metrics_multi_roles.csv found in {path}")

    if not datasets:
        raise SystemExit("No valid data found in the specified paths.")

    # 定义要绘制的图表和对应的列名
    plots_to_generate = [
        ("SU Success Rate", ["surate", "su_rate", "su_success_rate"], "Success Rate"),
        ("Throughput (Normalized)", ["throughput", "avg_throughput"], "Throughput per user per slot"),
        ("Collision Rate", ["coll_rate", "collision_rate"], "Collision Rate"),
        ("PU Success Rate", ["purate", "pu_rate", "pu_success_rate"], "Success Rate"),
        ("Total Return", ["return", "episode_return"], "Cumulative Reward"),
    ]

    print(f"\nGenerating {len(plots_to_generate)} comparison plots...")
    for title, cols, ylabel in plots_to_generate:
        plot_comparison(datasets, cols, title, ylabel)

    plt.show()


if __name__ == "__main__":
    main()