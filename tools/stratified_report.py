"""PRMI 分层评估报表生成器（顶刊可复现产物）。

输入：evaluate_epoch 生成的 per_frame_metrics.csv
输出：
  - stratified_report.csv：按 (subset, split, seq_len_bin, dt_bin, mixed) 汇总
  - stratified_summary.md ：可直接复制进论文附录

分桶规则严格对齐：
  - seq_len：1 / 2-3 / 4-6 / >=7
  - Δt(dt_prev)：<=30 / 30-90 / >90 / NA
  - mixed：flips>=1
"""

from __future__ import annotations

import os
import csv
import math
import argparse
from collections import defaultdict


def bin_seq_len(n: int) -> str:
    n = int(n)
    if n <= 1:
        return "1"
    if 2 <= n <= 3:
        return "2-3"
    if 4 <= n <= 6:
        return "4-6"
    return ">=7"


def bin_dt(dt: float) -> str:
    try:
        dt = float(dt)
    except Exception:
        return "NA"
    if dt >= 900:  # 约定：首帧 dt_prev=999.0
        return "NA"
    if dt <= 30:
        return "<=30"
    if dt <= 90:
        return "30-90"
    return ">90"


def _to_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _to_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def build_stratified_report(per_frame_csv: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # 需要聚合的核心指标
    mean_metrics = [
        "iou",
        "dice",
        "precision",
        "recall",
        "cldice",
        "sdf_relaxed_iou",
    ]

    # group_key -> accum
    acc = defaultdict(lambda: {
        "n_frames": 0,
        "seq_ids": set(),
        "sum": defaultdict(float),
        "sum_soil_fp": 0.0,
        "sum_soil_total": 0.0,
        "sum_tepr_soil": 0.0,
        "n_tepr_soil": 0,
    })

    with open(per_frame_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            subset = r.get("subset", "")
            split = r.get("split", "")
            seq_id = r.get("seq_id", "")
            seq_len = _to_int(r.get("seq_len", 0))
            dt = _to_float(r.get("delta_t", 999.0))
            mixed = r.get("mixed", "False")
            mixed = (str(mixed).lower() in ["1", "true", "yes"])

            seq_len_bin = r.get("seq_len_bin", "") or bin_seq_len(seq_len)
            dt_bin = r.get("dt_bin", "") or bin_dt(dt)

            key = (subset, split, seq_len_bin, dt_bin, str(mixed))
            a = acc[key]
            a["n_frames"] += 1
            a["seq_ids"].add(seq_id)
            for m in mean_metrics:
                a["sum"][m] += _to_float(r.get(m, 0.0))

            # 纯土期 FPR（像素级）
            gt_pixels = _to_int(r.get("gt_pixels", 0))
            if gt_pixels == 0:
                a["sum_soil_fp"] += _to_float(r.get("fp", 0.0))
                a["sum_soil_total"] += _to_float(r.get("total_pixels", 0.0))

                tepr = r.get("tepr_delta_fp_pct", "")
                if tepr != "":
                    a["sum_tepr_soil"] += _to_float(tepr, 0.0)
                    a["n_tepr_soil"] += 1

    out_csv = os.path.join(out_dir, "stratified_report.csv")
    fieldnames = [
        "subset", "split", "seq_len_bin", "dt_bin", "mixed",
        "n_frames", "n_sequences",
        *[f"mean_{m}" for m in mean_metrics],
        "pure_soil_fpr_pct",
        "mean_tepr_soil_pct",
    ]

    rows = []
    for (subset, split, seq_len_bin, dt_bin, mixed), a in sorted(acc.items()):
        n_frames = a["n_frames"]
        n_seq = len(a["seq_ids"])
        row = {
            "subset": subset,
            "split": split,
            "seq_len_bin": seq_len_bin,
            "dt_bin": dt_bin,
            "mixed": mixed,
            "n_frames": n_frames,
            "n_sequences": n_seq,
        }
        for m in mean_metrics:
            row[f"mean_{m}"] = (a["sum"][m] / max(1, n_frames))

        # soil fpr
        if a["sum_soil_total"] > 0:
            row["pure_soil_fpr_pct"] = 100.0 * a["sum_soil_fp"] / a["sum_soil_total"]
        else:
            row["pure_soil_fpr_pct"] = 0.0

        if a["n_tepr_soil"] > 0:
            row["mean_tepr_soil_pct"] = a["sum_tepr_soil"] / a["n_tepr_soil"]
        else:
            row["mean_tepr_soil_pct"] = 0.0

        rows.append(row)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown summary（附录即插即用）
    md_path = os.path.join(out_dir, "stratified_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PRMI Stratified Evaluation Summary\n\n")
        f.write(f"Input CSV: `{os.path.basename(per_frame_csv)}`\n\n")
        f.write("## Buckets\n")
        f.write("- seq_len: 1 / 2-3 / 4-6 / >=7\n")
        f.write("- Δt(dt_prev): <=30 / 30-90 / >90 / NA\n")
        f.write("- mixed: flips>=1\n\n")
        f.write("## Output\n")
        f.write(f"- `stratified_report.csv`\n")
        f.write(f"- `stratified_summary.md`\n\n")
        f.write("> 说明：mean_* 指标为逐帧平均；Pure_Soil_FPR 为纯土帧像素级 FP/Total。\n")

    return {"out_csv": out_csv, "out_md": md_path, "n_groups": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_frame_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()
    res = build_stratified_report(args.per_frame_csv, args.out_dir)
    print(f"✅ Stratified report saved: {res['out_csv']} | groups={res['n_groups']}")
    print(f"✅ Markdown summary saved: {res['out_md']}")


if __name__ == "__main__":
    main()
