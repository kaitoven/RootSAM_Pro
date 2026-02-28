import os
import argparse
import pandas as pd


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _agg_table(df: pd.DataFrame, group_cols, metric_cols):
    g = df.groupby(group_cols, dropna=False)
    out = g.size().rename("n_frames").to_frame()
    for c in metric_cols:
        if c in df.columns:
            out[c] = g[c].mean()
    out = out.reset_index()
    return out


def main():
    ap = argparse.ArgumentParser(description="Rollout stratified report (PRMI protocol)")
    ap.add_argument("--per_frame_csv", type=str, required=True, help="Path to per_frame_metrics.csv produced by engine.trainer.evaluate_epoch")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir for stratified CSV tables")
    args = ap.parse_args()

    df = pd.read_csv(args.per_frame_csv)

    # Canonicalize columns
    if "dt_bin" not in df.columns and "delta_t" in df.columns:
        # fallback binning
        def _bin_dt(x):
            try:
                x = float(x)
                if x >= 998.0:
                    return "NA"
            except Exception:
                return "NA"
            if x <= 30:
                return "<=30"
            if x <= 90:
                return "30-90"
            return ">90"
        df["dt_bin"] = df["delta_t"].apply(_bin_dt)

    if "seq_len_bin" not in df.columns and "seq_len" in df.columns:
        def _bin_len(L):
            try:
                L = int(L)
            except Exception:
                return "NA"
            if L <= 1:
                return "1"
            if L <= 3:
                return "2-3"
            if L <= 6:
                return "4-6"
            return ">=7"
        df["seq_len_bin"] = df["seq_len"].apply(_bin_len)

    if "mixed" not in df.columns and "flips" in df.columns:
        df["mixed"] = df["flips"].fillna(0).astype(int) >= 1

    # Only evaluate frames that have GT info
    metric_cols = [c for c in ["iou", "dice", "precision", "recall", "cldice", "sdf_relaxed_iou"] if c in df.columns]

    _ensure_dir(args.out_dir)

    # Overall
    overall = pd.DataFrame([{
        "n_frames": int(len(df)),
        **{c: float(df[c].mean()) for c in metric_cols}
    }])
    overall.to_csv(os.path.join(args.out_dir, "overall.csv"), index=False)

    # Stratified: seq_len bins
    t_seq = _agg_table(df, ["seq_len_bin"], metric_cols)
    t_seq.to_csv(os.path.join(args.out_dir, "strat_seq_len.csv"), index=False)

    # Stratified: dt bins (exclude NA like PRMI protocol)
    df_dt = df[df["dt_bin"].astype(str) != "NA"].copy()
    t_dt = _agg_table(df_dt, ["dt_bin"], metric_cols)
    t_dt.to_csv(os.path.join(args.out_dir, "strat_dt.csv"), index=False)

    # Stratified: static vs dynamic
    t_dyn = _agg_table(df, ["mixed"], metric_cols)
    # rename to match doc terminology
    t_dyn["dynamic"] = t_dyn["mixed"].apply(lambda x: "dynamic" if bool(x) else "static")
    t_dyn = t_dyn.drop(columns=["mixed"])
    t_dyn.to_csv(os.path.join(args.out_dir, "strat_dynamic.csv"), index=False)

    # (Optional) subset×split summary if columns exist
    if "subset" in df.columns and "split" in df.columns:
        t_ss = _agg_table(df, ["subset", "split"], metric_cols)
        t_ss.to_csv(os.path.join(args.out_dir, "subset_split.csv"), index=False)

    print("[OK] wrote:", args.out_dir)


if __name__ == "__main__":
    main()
