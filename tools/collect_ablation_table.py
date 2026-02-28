#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust ablation table collector for RootSAM-Pro.

Why this exists:
- Some runs may not produce report/test_summary.csv (e.g., test disabled, or only test_runs.csv exists).
- We still want to collect runs deterministically and produce Table 4.1 outputs.
- We also want to ignore non-run folders like ablation_tables itself.

Inputs (expected per run):
  <run_dir>/report/run_manifest.json
  and one of:
    <run_dir>/report/test_summary.csv               (preferred if --prefer test)
    <run_dir>/report/test_runs.csv                  (fallback)
    <run_dir>/report/best_summary.csv               (fallback or preferred if --prefer best)

Outputs:
  <runs_root>/ablation_tables/ablation_runs.csv
  <runs_root>/ablation_tables/Table_4_1_mean_std.csv
  <runs_root>/ablation_tables/Table_4_1_best_seed.csv
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

NUMERIC_COLS_DEFAULT = [
    "Accuracy", "Precision", "Recall", "F1_Score", "Standard_IoU", "SDF_Relaxed_IoU",
    "clDice", "Pure_Soil_FPR", "Insular_Gap_Recall", "Gap_mIoU",
    "HPACS_SCORE", "BEST_SCORE"
]

def _to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")

def _read_csv_one_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
        # summary files should be single-row; if multiple, take last
        return rows[-1]

def _read_test_runs_pick_best(path: Path) -> Optional[Dict[str, Any]]:
    """Fallback: parse test_runs.csv and pick BEST_SCORE if present, else last row."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
    # Try to find a row tagged BEST_SCORE / BEST_HPACS / LAST
    tag_keys = ["which", "tag", "ckpt_tag", "name"]
    def get_tag(r):
        for k in tag_keys:
            if k in r and r[k]:
                return str(r[k])
        return ""
    # prefer BEST_SCORE row if exists
    for r in rows:
        if "BEST_SCORE" in get_tag(r):
            return r
    # else prefer LAST
    for r in rows:
        if "LAST" in get_tag(r):
            return r
    return rows[-1]

def _load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    mf = run_dir / "report" / "run_manifest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None

def _get_manifest_field(man: Dict[str, Any], *keys: str, default=None):
    cur = man
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _resolve_summary(run_dir: Path, prefer: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (row, source)."""
    report = run_dir / "report"
    test_summary = report / "test_summary.csv"
    best_summary = report / "best_summary.csv"
    test_runs = report / "test_runs.csv"

    if prefer == "test":
        row = _read_csv_one_row(test_summary)
        if row is not None:
            return row, "test_summary"
        row = _read_test_runs_pick_best(test_runs)
        if row is not None:
            return row, "test_runs"
        row = _read_csv_one_row(best_summary)
        if row is not None:
            return row, "best_summary_fallback"
        return None, "missing"

    # prefer == best
    row = _read_csv_one_row(best_summary)
    if row is not None:
        return row, "best_summary"
    row = _read_csv_one_row(test_summary)
    if row is not None:
        return row, "test_summary_fallback"
    row = _read_test_runs_pick_best(test_runs)
    if row is not None:
        return row, "test_runs_fallback"
    return None, "missing"

def _mean_std(vals: List[float]) -> Tuple[float, float]:
    xs = [v for v in vals if not math.isnan(v)]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)

def _is_run_dir(p: Path) -> bool:
    # Ignore known non-run dirs
    if not p.is_dir():
        return False
    if p.name in {"ablation_tables", "logs", "viz", "report"}:
        return False
    # A run dir should have report/run_manifest.json
    return (p / "report" / "run_manifest.json").exists()

def _pick_best_run(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deterministic best selection per (subset,ablation)."""
    if not rows:
        return None

    def key(r: Dict[str, Any]):
        # Prefer HPACS_SCORE if available, else BEST_SCORE, else F1/IoU/clDice/FPR
        hpacs = _to_float(r.get("HPACS_SCORE", float("nan")))
        best_score = _to_float(r.get("BEST_SCORE", float("nan")))
        f1 = _to_float(r.get("F1_Score"))
        iou = _to_float(r.get("Standard_IoU"))
        cld = _to_float(r.get("clDice"))
        fpr = _to_float(r.get("Pure_Soil_FPR"))
        # Sorting: higher is better for hpacs/best_score/f1/iou/cld; lower is better for fpr
        # Use -value for descending, +fpr for ascending
        # If hpacs is NaN, treat as very small
        hpacs_key = -hpacs if not math.isnan(hpacs) else float("inf")
        best_score_key = -best_score if not math.isnan(best_score) else float("inf")
        return (hpacs_key, best_score_key, -f1, -iou, -cld, fpr)

    return sorted(rows, key=key)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True, help="Root directory containing run folders")
    ap.add_argument("--subset", type=str, default=None, help="Filter by subset name")
    ap.add_argument("--ablation", type=str, default=None, help="Filter by ablation mode")
    ap.add_argument("--prefer", type=str, default="test", choices=["test", "best"], help="Prefer test_summary or best_summary")
    ap.add_argument("--strict", action="store_true", help="Error if any run is missing manifest/summary")
    ap.add_argument("--progress_every", type=int, default=50, help="Print progress every N scanned dirs")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    out_dir = runs_root / "ablation_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[collect_ablation_table] runs_root = {runs_root}")
    print(f"[collect_ablation_table] out_dir   = {out_dir}")

    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    collected: List[Dict[str, Any]] = []
    skipped: List[str] = []

    scanned = 0
    for run_dir in sorted(run_dirs):
        scanned += 1
        if args.progress_every > 0 and scanned % args.progress_every == 0:
            print(f"[collect_ablation_table] scanning {scanned}/{len(run_dirs)} ...")

        if not _is_run_dir(run_dir):
            # ignore silently
            continue

        man = _load_manifest(run_dir)
        if man is None:
            msg = f"missing manifest: {run_dir}"
            if args.strict:
                raise RuntimeError(msg)
            skipped.append(msg)
            continue

        subset = _get_manifest_field(man, "cfg", "SUBSET", default=_get_manifest_field(man, "args", "subset", default=""))
        ablation = _get_manifest_field(man, "cfg", "ABLATION_MODE", default=_get_manifest_field(man, "args", "ablation", default=""))
        seed = _get_manifest_field(man, "cfg", "SEED", default=_get_manifest_field(man, "args", "SEED", default=""))

        if args.subset and str(subset) != args.subset:
            continue
        if args.ablation and str(ablation) != args.ablation:
            continue

        row, source = _resolve_summary(run_dir, args.prefer)
        if row is None:
            msg = f"missing summary ({args.prefer}): {run_dir}"
            if args.strict:
                raise RuntimeError(msg)
            skipped.append(msg)
            continue

        # Build record
        rec: Dict[str, Any] = {}
        rec["run_dir"] = str(run_dir)
        rec["subset"] = str(subset)
        rec["ablation"] = str(ablation)
        rec["seed"] = str(seed)
        rec["summary_source"] = source

        # bring numeric cols if present
        for k, v in row.items():
            rec[k] = v

        # also bring a few useful manifest fields
        rec["run_id"] = _get_manifest_field(man, "run", "run_id", default=_get_manifest_field(man, "args", "run_id", default=""))
        rec["version"] = _get_manifest_field(man, "run", "version", default=_get_manifest_field(man, "version", default=""))
        rec["code_hash"] = _get_manifest_field(man, "run", "code_hash", default=_get_manifest_field(man, "code", "code_hash_short", default=""))

        collected.append(rec)

    print(f"[collect_ablation_table] collected = {len(collected)} runs")
    print(f"[collect_ablation_table] skipped  = {len(skipped)} (use --strict to error)")
    if skipped:
        for s in skipped[:20]:
            print("  - " + s)
        if len(skipped) > 20:
            print(f"  ... ({len(skipped)-20} more)")

    # Write ablation_runs.csv
    if collected:
        # Determine columns
        cols = ["subset","ablation","seed","run_id","version","code_hash","summary_source","run_dir"]
        # Add metric columns present in any row
        metric_cols = []
        for rec in collected:
            for k in rec.keys():
                if k in cols:
                    continue
                if k not in metric_cols and k not in {"run_dir","subset","ablation","seed","run_id","version","code_hash","summary_source"}:
                    metric_cols.append(k)
        # Stable ordering: preferred numeric cols first then rest
        numeric_pref = [c for c in NUMERIC_COLS_DEFAULT if c in metric_cols]
        rest = [c for c in metric_cols if c not in numeric_pref]
        cols = cols + numeric_pref + sorted(rest)

        runs_csv = out_dir / "ablation_runs.csv"
        with runs_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for rec in collected:
                w.writerow({k: rec.get(k, "") for k in cols})

        # Group by subset, ablation
        groups: Dict[Tuple[str,str], List[Dict[str, Any]]] = {}
        for rec in collected:
            groups.setdefault((rec["subset"], rec["ablation"]), []).append(rec)

        # mean±std table
        mean_std_rows: List[Dict[str, Any]] = []
        best_rows: List[Dict[str, Any]] = []

        # Decide numeric columns to summarize: intersection with collected keys
        all_keys = set().union(*(r.keys() for r in collected))
        num_cols = [c for c in NUMERIC_COLS_DEFAULT if c in all_keys]
        # also add common metrics if present
        for extra in ["Loss", "Score"]:
            if extra in all_keys and extra not in num_cols:
                num_cols.append(extra)

        for (subset, ablation), rs in sorted(groups.items()):
            out = {"subset": subset, "ablation": ablation, "n": str(len(rs))}
            for c in num_cols:
                vals = [_to_float(r.get(c)) for r in rs]
                m, s = _mean_std(vals)
                out[c] = f"{m:.4f}±{s:.4f}" if not math.isnan(m) else ""
                out[c+"_mean"] = f"{m:.6f}" if not math.isnan(m) else ""
                out[c+"_std"] = f"{s:.6f}" if not math.isnan(s) else ""
            mean_std_rows.append(out)

            best = _pick_best_run(rs)
            if best is not None:
                best_rows.append(best)

        # write mean/std
        mean_csv = out_dir / "Table_4_1_mean_std.csv"
        # columns: subset, ablation, n, then each metric (pretty), and also mean/std raw columns
        mean_cols = ["subset","ablation","n"]
        for c in num_cols:
            mean_cols.append(c)
        # keep raw mean/std for plotting
        for c in num_cols:
            mean_cols.append(c+"_mean")
            mean_cols.append(c+"_std")

        with mean_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=mean_cols)
            w.writeheader()
            for r in mean_std_rows:
                w.writerow({k: r.get(k,"") for k in mean_cols})

        # write best seed table (use same columns as ablation_runs but only best per group)
        best_csv = out_dir / "Table_4_1_best_seed.csv"
        with best_csv.open("w", newline="", encoding="utf-8") as f:
            # reuse columns from runs_csv
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in best_rows:
                w.writerow({k: r.get(k, "") for k in cols})

        print(f"[collect_ablation_table] wrote: {runs_csv}")
        print(f"[collect_ablation_table] wrote: {mean_csv}")
        print(f"[collect_ablation_table] wrote: {best_csv}")
    else:
        # still create empty placeholder
        (out_dir / "ablation_runs.csv").write_text("", encoding="utf-8")

if __name__ == "__main__":
    main()
