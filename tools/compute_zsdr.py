"""ZS-DR (Zero-shot Domain Retention) 计算工具。

根据《评价指标》：
  ZS-DR = Acc_ft / Acc_zero_shot

这里的 Acc 可以选择：F1(Dice) / IoU / (SDF-relaxed IoU + clDice) 等。

用法示例：
  python tools/compute_zsdr.py --ft_csv runs/DomainA_FULL_report/per_frame_metrics.csv \
                              --zs_csv runs/DomainA_ZERO_SHOT_report/per_frame_metrics.csv \
                              --metric sdf_relaxed_iou --metric2 cldice

如果同时提供 metric2，则使用 (metric + metric2) 的均值作为 Acc。
"""

from __future__ import annotations

import csv
import argparse


def _mean_metric(csv_path: str, metric: str, metric2: str | None = None) -> float:
    s = 0.0
    n = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                v1 = float(r.get(metric, 0.0))
                if metric2:
                    v2 = float(r.get(metric2, 0.0))
                    v = 0.5 * (v1 + v2)
                else:
                    v = v1
                s += v
                n += 1
            except Exception:
                continue
    return s / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft_csv", type=str, required=True, help="Fine-tuned per_frame_metrics.csv")
    ap.add_argument("--zs_csv", type=str, required=True, help="Zero-shot per_frame_metrics.csv")
    ap.add_argument("--metric", type=str, default="sdf_relaxed_iou")
    ap.add_argument("--metric2", type=str, default=None)
    args = ap.parse_args()

    ft = _mean_metric(args.ft_csv, args.metric, args.metric2)
    zs = _mean_metric(args.zs_csv, args.metric, args.metric2)
    zsdr = ft / max(1e-8, zs)

    name = args.metric if not args.metric2 else f"0.5*({args.metric}+{args.metric2})"
    print(f"FT mean {name}: {ft:.6f}")
    print(f"ZS mean {name}: {zs:.6f}")
    print(f"ZS-DR = FT/ZS : {zsdr:.6f}")


if __name__ == "__main__":
    main()
