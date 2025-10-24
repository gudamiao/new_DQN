#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training metrics from outputs/sc_*/metrics_multi_roles.csv

Usage:
  python plot_metrics.py outputs/sc_base
  # 若不传 path，会在当前目录下递归寻找最新的 metrics_multi_roles.csv
"""
# plot_metrics.py
import argparse
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

def find_latest_metrics(start_dir="."):
    candidates = []
    for root, _, files in os.walk(start_dir):
        for name in files:
            if name == "metrics_multi_roles.csv":
                fp = os.path.join(root, name)
                try:
                    mtime = os.path.getmtime(fp)
                except Exception:
                    mtime = 0
                candidates.append((mtime, fp))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def read_meta_if_any(csv_path):
    meta_path = os.path.join(os.path.dirname(csv_path), "run_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# --------- 新增：更健壮的列名匹配 ----------
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def get_col(df: pd.DataFrame, candidates):
    # candidates: 可能的列名列表（不区分大小写、会去掉非字母数字）
    want = [_norm(x) for x in candidates]
    mapping = { _norm(c): c for c in df.columns }
    for w in want:
        if w in mapping:
            return mapping[w]
    return None
# -----------------------------------------

def ensure_cols(df: pd.DataFrame, meta):
    # 兼容 Return/return
    if "return" not in df.columns and "Return" in df.columns:
        df.rename(columns={"Return": "return"}, inplace=True)

    # 碰撞率 coll_rate = collision / tx
    if "coll_rate" not in df.columns:
        coll = get_col(df, ["collision", "collisions"])
        tx   = get_col(df, ["tx", "attempts", "transmissions"])
        if coll and tx:
            df["coll_rate"] = df[coll] / df[tx].clip(lower=1)

    # Wait%：需要 meta 的 episode_len * num_users
    if "wait_percent" not in df.columns:
        wait_col = get_col(df, ["wait", "wait_slots", "waiting"])
        denom = None
        if meta and "episode_len" in meta and "num_users" in meta:
            denom = meta["episode_len"] * meta["num_users"]
        if wait_col and denom:
            df["wait_percent"] = df[wait_col] * 100.0 / float(denom)

    # Throughput：若没有，按 sum_throughput / (T*N)
    if "throughput" not in df.columns:
        sum_thr = get_col(df, ["sum_throughput", "sumthr", "sum_thr"])
        if sum_thr and meta and "episode_len" in meta and "num_users" in meta:
            denom = meta["episode_len"] * meta["num_users"]
            df["throughput"] = df[sum_thr] / float(denom)

    # Coll/100：若没有，按 collision / (T*N) * 100
    if get_col(df, ["coll/100", "coll_per_100", "collper100"]) is None:
        coll = get_col(df, ["collision", "collisions"])
        if coll and meta and "episode_len" in meta and "num_users" in meta:
            denom = meta["episode_len"] * meta["num_users"]
            df["coll_per_100"] = df[coll] * 100.0 / float(denom)

def plot_one(x, y, title, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)

def main():
    ap = argparse.ArgumentParser(description="Plot metrics for DQN training")
    ap.add_argument("path", nargs="?", default=None,
                    help="metrics_multi_roles.csv 的路径（或包含它的目录）。留空则自动搜索。")
    args = ap.parse_args()

    # 定位 CSV
    if args.path is None:
        csv_path = find_latest_metrics(".")
        if csv_path is None:
            raise SystemExit("未找到 metrics_multi_roles.csv，请先运行训练或手动指定路径。")
    else:
        if os.path.isdir(args.path):
            guess = os.path.join(args.path, "metrics_multi_roles.csv")
            csv_path = guess if os.path.isfile(guess) else find_latest_metrics(args.path)
        else:
            csv_path = args.path

    if csv_path is None or not os.path.isfile(csv_path):
        raise SystemExit("未找到有效的 metrics_multi_roles.csv。")

    meta = read_meta_if_any(csv_path)
    df = pd.read_csv(csv_path)
    ensure_cols(df, meta)

    # X 轴
    ep_col = get_col(df, ["episode", "ep", "iter", "epoch"])
    x = df[ep_col] if ep_col else range(1, len(df) + 1)

    # 需要画的指标名及候选列名列表（顺序越靠前优先级越高）
    targets = [
        ("Return",            ["return", "episode_return", "ret"],                 "return"),
        ("SuccRate",          ["succrate", "success_rate", "succ_rate"],           "rate"),
        ("SUrate",            ["surate", "su_rate", "su_success_rate"],            "rate"),
        ("PUrate",            ["purate", "pu_rate", "pu_success_rate"],            "rate"),
        ("Wait %",            ["wait_percent", "wait%", "wait_ratio", "wait_rate"],"percent"),
        ("Throughput", ["throughput", "avg_throughput", "thr"],
         "Throughput (b/s/Hz per user per slot)"),
        ("Coll/100",          ["coll/100", "coll_per_100", "collper100"],          "per 100 slots"),
    ]

    # 逐项画图（若缺列会自动跳过）
    for title, cands, ylab in targets:
        col = get_col(df, cands)
        if col is None and title == "Coll/100":
            # 兼容 ensure_cols 自动生成的 coll_per_100
            if "coll_per_100" in df.columns:
                col = "coll_per_100"
        if col is not None:
            plot_one(x, df[col], title, ylab)
        else:
            print(f"[WARN] 跳过 {title}：未找到列 {cands}")

    plt.show()

if __name__ == "__main__":
    main()
