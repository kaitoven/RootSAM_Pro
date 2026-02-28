#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot PP-FSRD++ (polar-band Fourier SRD) statistics saved in train_val_history.csv.

This script is *purely for visualization* and does not affect training.

Typical usage:
  python -m RootSAM_Pro.tools.plot_ppfsrd_curves \
    --csv runs_tl_papaya_mix_amp/report/train_val_history.csv \
    --out_dir runs_tl_papaya_mix_amp/report/plots_ppfsrd

Outputs:
  - ppfsrd_gamma.png/.svg
  - ppfsrd_polar_weight.png/.svg
  - ppfsrd_radial_bands.png/.svg
  - ppfsrd_orient_bands.png/.svg
  - ppfsrd_radial_heatmap.png/.svg
  - ppfsrd_orient_heatmap.png/.svg
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAD_PREFIX = "PPFSRD_rad_band_abs_mean_"
ORI_PREFIX = "PPFSRD_ori_band_abs_mean_"


@dataclass
class SeriesPack:
    epoch: np.ndarray
    cols: Dict[str, np.ndarray]


def _read_csv_as_pack(path: str) -> SeriesPack:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    if pd is not None:
        df = pd.read_csv(path)
        # Find an epoch column (be tolerant)
        epoch_col = None
        for c in ["epoch", "Epoch", "ep"]:
            if c in df.columns:
                epoch_col = c
                break
        if epoch_col is None:
            # fallback to row index
            epoch = np.arange(len(df), dtype=np.float32)
        else:
            epoch = df[epoch_col].to_numpy(dtype=np.float32)

        cols: Dict[str, np.ndarray] = {}
        for c in df.columns:
            if c == epoch_col:
                continue
            # keep only numeric columns
            try:
                cols[c] = df[c].to_numpy(dtype=np.float32)
            except Exception:
                continue

        return SeriesPack(epoch=epoch, cols=cols)

    # csv module fallback
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    # epoch
    epoch = []
    for r in rows:
        if "epoch" in r and r["epoch"] != "":
            epoch.append(float(r["epoch"]))
        else:
            epoch.append(float(len(epoch)))
    epoch = np.asarray(epoch, dtype=np.float32)

    cols: Dict[str, np.ndarray] = {}
    keys = rows[0].keys()
    for k in keys:
        if k == "epoch":
            continue
        vals = []
        ok = True
        for r in rows:
            try:
                vals.append(float(r.get(k, "nan")))
            except Exception:
                ok = False
                break
        if ok:
            cols[k] = np.asarray(vals, dtype=np.float32)

    return SeriesPack(epoch=epoch, cols=cols)


def _ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def _save_fig(out_dir: str, stem: str) -> None:
    png = os.path.join(out_dir, f"{stem}.png")
    svg = os.path.join(out_dir, f"{stem}.svg")
    plt.savefig(png, bbox_inches="tight", dpi=180)
    plt.savefig(svg, bbox_inches="tight")


def _plot_line(x: np.ndarray, ys: List[Tuple[str, np.ndarray]], title: str, xlabel: str, ylabel: str) -> None:
    plt.figure()
    for name, y in ys:
        if y is None:
            continue
        plt.plot(x, y, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(ys) > 1:
        plt.legend(loc="best")


def _plot_heatmap(x: np.ndarray, band_names: List[str], band_series: List[np.ndarray], title: str, xlabel: str, ylabel: str) -> None:
    # band_series: list of (T,) arrays; stack => (K,T)
    if not band_series:
        return
    mat = np.stack(band_series, axis=0)

    plt.figure()
    # Use imshow with nearest neighbor to keep it simple
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # tick labels: keep light
    # x-axis ticks (epoch) sparse
    if len(x) > 1:
        xt = np.linspace(0, len(x) - 1, num=min(6, len(x)), dtype=int)
        plt.xticks(xt, [str(int(x[i])) for i in xt])

    yt = np.arange(len(band_names))
    plt.yticks(yt, band_names)

    plt.colorbar()


def _extract_band_series(pack: SeriesPack, prefix: str) -> Tuple[List[str], List[np.ndarray]]:
    names = [c for c in pack.cols.keys() if c.startswith(prefix)]
    # sort by k index if present
    def _key(c: str) -> int:
        tail = c[len(prefix):]
        # expected like k0, k1 ...
        if tail.startswith("k"):
            try:
                return int(tail[1:])
            except Exception:
                return 10**9
        try:
            return int(tail)
        except Exception:
            return 10**9

    names = sorted(names, key=_key)
    series = [pack.cols[n] for n in names]
    # shorten labels: k0..k5
    labels = [n[len(prefix):] for n in names]
    return labels, series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to train_val_history.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory for plots")
    ap.add_argument("--title_prefix", default="", help="Optional prefix for figure titles")
    args = ap.parse_args()

    pack = _read_csv_as_pack(args.csv)
    _ensure_out_dir(args.out_dir)

    prefix = (args.title_prefix + " ") if args.title_prefix else ""

    # 1) gamma strength
    gamma = pack.cols.get("PPFSRD_gamma_abs_mean", None)
    if gamma is not None:
        _plot_line(pack.epoch, [("|gamma| mean", gamma)], f"{prefix}PP-FSRD++ Fourier Residual Strength", "epoch", "mean(|gamma|)")
        _save_fig(args.out_dir, "ppfsrd_gamma")
        plt.close()

    # 2) polar weight stats
    w_mean = pack.cols.get("PPFSRD_polar_w_abs_mean", None)
    w_max = pack.cols.get("PPFSRD_polar_w_abs_max", None)
    ys = []
    if w_mean is not None:
        ys.append(("mean(|w|)", w_mean))
    if w_max is not None:
        ys.append(("max(|w|)", w_max))
    if ys:
        _plot_line(pack.epoch, ys, f"{prefix}PP-FSRD++ Polar Band Weights", "epoch", "abs(weight)")
        _save_fig(args.out_dir, "ppfsrd_polar_weight")
        plt.close()

    # 3) radial bands (lines)
    rad_labels, rad_series = _extract_band_series(pack, RAD_PREFIX)
    if rad_series:
        _plot_line(pack.epoch, list(zip(rad_labels, rad_series)), f"{prefix}PP-FSRD++ Radial Band Marginals", "epoch", "mean(|w|) per radial band")
        _save_fig(args.out_dir, "ppfsrd_radial_bands")
        plt.close()

        _plot_heatmap(pack.epoch, rad_labels, rad_series, f"{prefix}PP-FSRD++ Radial Bands Heatmap", "epoch", "radial band")
        _save_fig(args.out_dir, "ppfsrd_radial_heatmap")
        plt.close()

    # 4) orientation bands (lines)
    ori_labels, ori_series = _extract_band_series(pack, ORI_PREFIX)
    if ori_series:
        _plot_line(pack.epoch, list(zip(ori_labels, ori_series)), f"{prefix}PP-FSRD++ Orientation Band Marginals", "epoch", "mean(|w|) per orientation band")
        _save_fig(args.out_dir, "ppfsrd_orient_bands")
        plt.close()

        _plot_heatmap(pack.epoch, ori_labels, ori_series, f"{prefix}PP-FSRD++ Orientation Bands Heatmap", "epoch", "orientation band")
        _save_fig(args.out_dir, "ppfsrd_orient_heatmap")
        plt.close()

    # 5) write a tiny text summary for convenience
    summary_path = os.path.join(args.out_dir, "ppfsrd_plot_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"csv: {args.csv}\n")
        if gamma is not None:
            f.write(f"PPFSRD_gamma_abs_mean: last={float(gamma[-1]):.6f}\n")
        if w_mean is not None:
            f.write(f"PPFSRD_polar_w_abs_mean: last={float(w_mean[-1]):.6f}\n")
        if w_max is not None:
            f.write(f"PPFSRD_polar_w_abs_max: last={float(w_max[-1]):.6f}\n")
        f.write(f"radial bands: {len(rad_series)}\n")
        f.write(f"orient bands: {len(ori_series)}\n")


if __name__ == "__main__":
    main()
