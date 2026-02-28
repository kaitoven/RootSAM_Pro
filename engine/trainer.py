import os
import csv
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from engine.metrics import BeyondGT_MetricsTracker, compute_frame_metrics
from utils.transforms import PhysicalPreservingTransforms


def _get_amp_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


def train_epoch(model, dataloader, optimizer, scaler, criterion, device, cfg):
    """Truncated-BPTT training over fixed-length clips.

    - Checkpoint-safe: model forward returns NEW state; outer loop only does pointer handoff.
    - Compatible with optional 6th return (gate_loss).
    """
    model.train()
    epoch_loss = 0.0

    # Component loss meters (epoch averages)
    comp_sum = {
        'Total': 0.0,
        'Seg_Exempt': 0.0,
        'Topo_clDice': 0.0,
        'Probe': 0.0,
        'Kinematic': 0.0,
        'Presence': 0.0,
        'KeyframeGate': 0.0,
    }
    comp_count = 0


    # OOM safety: allow gradients only for explicitly trainable modules
    for name, param in model.named_parameters():
        if param.requires_grad:
            allowed = (
                ("srd" in name) or ("kmr" in name) or ("kappa" in name)
                or ("temporal_mlp" in name) or ("keyframe_mlp" in name) or ("key_in_norm" in name)
                # InfoGain value head (small, safe trainables)
                or ("value_in_norm" in name) or ("value_mlp" in name) or ("value_half_life" in name)
                or ("bio_half_life" in name) or ("ptr_half_life" in name)
                or ("pres_" in name) or ("abs_bias" in name)
                or ("eta_energy" in name) or ("theta_frag" in name)
                or ("key_temp" in name) or ("key_log_vars" in name)
                or ("resource_lambda" in name)
                or ("bhfi" in name)
                or ("sfa" in name)
                or ("pra" in name)
                or ("asta" in name)
                or (".mlp.adapter." in name)
            )
            assert allowed, f"OOM Danger! Parameter {name} leaked gradient."

    use_amp = bool(getattr(cfg, "AMP", True))
    amp_dtype = _get_amp_dtype(device)
    non_blocking = bool(getattr(cfg, "NON_BLOCKING", True))

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device, non_blocking=non_blocking)
        masks_gt = batch["masks_gt"].to(device, non_blocking=non_blocking)
        delta_ts = batch["delta_t"].to(device, non_blocking=non_blocking)
        W_SDF = batch["W_SDF"].to(device, non_blocking=non_blocking)
        pad_infos = batch["pad_info"].to(device, non_blocking=non_blocking)
        metas = batch.get("meta", None)

        B, T = images.shape[:2]

        # --- Safety: sequence coherence (no mixing frames across seq_id within a sample) ---
        if bool(getattr(cfg, "ASSERT_SEQ_COHERENCE", True)) and (metas is not None):
            def _get_meta_train(metas_obj, t_idx: int, b_idx: int) -> dict:
                if metas_obj is None:
                    return {}
                try:
                    mt = metas_obj[t_idx]
                except Exception:
                    return {}
                if isinstance(mt, list):
                    return mt[b_idx] if b_idx < len(mt) else {}
                if isinstance(mt, dict):
                    out = {}
                    for k, v in mt.items():
                        try:
                            out[k] = v[b_idx]
                        except Exception:
                            out[k] = v
                    return out
                return {}
            for b in range(B):
                sids = set()
                for t in range(T):
                    mt = _get_meta_train(metas, t, b)
                    sid = str(mt.get("seq_id", "")).strip()
                    if sid:
                        sids.add(sid)
                if len(sids) > 1:
                    raise RuntimeError(f"[SeqCoherenceError] Mixed seq_id within one sample b={b}: {sorted(list(sids))[:6]}")

        # v6+ functional state init (Dual FIFO + time axis + presence memory)
        inference_state = {}
        F_track_prev = None
        total_loss = None
        last_loss_dict = None

        train_mode = str(getattr(cfg, "TRAIN_MODE", "CLIP")).upper()
        tbptt_chunk = int(getattr(cfg, "TBPTT_CHUNK", 0) or 0)
        use_tbptt_chunk = (train_mode == "SEQUENCE" and tbptt_chunk > 0)

        def _detach_state(x):
            if torch.is_tensor(x):
                return x.detach()
            if isinstance(x, dict):
                return {k: _detach_state(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_detach_state(v) for v in x]
            return x

        optimizer.zero_grad(set_to_none=True)

        # For CLIP mode we keep the original "one step per batch" behavior.
        total_loss = None
        chunk_loss = None
        chunk_count = 0

        # batch telemetry (average over valid frames)
        batch_loss_sum = 0.0
        batch_frames = 0
        batch_comp = {k: 0.0 for k in comp_sum.keys()}

        for t in range(T):
            img_t = images[:, t]
            dt_t = delta_ts[:, t]
            pad_t = pad_infos[:, t]

            # checkpoint pacemaker
            img_t_safe = img_t.detach().requires_grad_(True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                out = checkpoint.checkpoint(model, img_t_safe, dt_t, inference_state, t, use_reentrant=False)

                gate_loss = None
                if isinstance(out, (tuple, list)) and len(out) >= 6:
                    M_fused_logits, probes_prob, F_kin, F_track, new_state, gate_loss = out[:6]
                else:
                    M_fused_logits, probes_prob, F_kin, F_track, new_state = out[:5]

                # handoff (functional state)
                inference_state = new_state

                # print debug events if any (limit spam)
                dbg = None
                if isinstance(inference_state, dict) and "__debug__" in inference_state:
                    dbg = inference_state.pop("__debug__", None)
                if dbg is not None and step < 2 and t < 2:
                    print(f"[DBG][train][step={step}][t={t}] {dbg}")

                # logits -> 1024 -> physical
                M_fused_1024 = F.interpolate(
                    M_fused_logits.float(),
                    size=(cfg.TARGET_SIZE, cfg.TARGET_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                M_fused_phys = torch.cat(
                    [PhysicalPreservingTransforms.reverse_logits_to_physical(M_fused_1024[b:b + 1], pad_t[b]) for b in range(B)],
                    dim=0,
                )

                probes_1024 = F.interpolate(
                    probes_prob.float(),
                    size=(cfg.TARGET_SIZE, cfg.TARGET_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                probes_phys = torch.cat(
                    [PhysicalPreservingTransforms.reverse_logits_to_physical(probes_1024[b:b + 1], pad_t[b]) for b in range(B)],
                    dim=0,
                )

                Pc_gt = batch["P_c"][:, t].to(device, non_blocking=non_blocking)
                Pt_gt = batch["P_t"][:, t].to(device, non_blocking=non_blocking)
                Ph_gt = batch["P_h"][:, t].to(device, non_blocking=non_blocking)

                loss_t, loss_dict = criterion(
                    M_fused_phys, masks_gt[:, t], W_SDF[:, t],
                    probes_phys, Pc_gt, Pt_gt, Ph_gt,
                    F_track, F_track_prev, F_kin,
                )

                if gate_loss is not None and torch.is_tensor(gate_loss):
                    loss_t = loss_t + gate_loss
                    loss_dict = dict(loss_dict)
                    loss_dict["KeyframeGate"] = float(gate_loss.detach().item())

                # telemetry
                batch_loss_sum += float(loss_t.detach().item())
                batch_frames += 1
                for k in batch_comp.keys():
                    try:
                        batch_comp[k] += float(loss_dict.get(k, 0.0))
                    except Exception:
                        pass

                if use_tbptt_chunk:
                    chunk_loss = loss_t if chunk_loss is None else (chunk_loss + loss_t)
                    chunk_count += 1
                    if (chunk_count >= tbptt_chunk) or (t == T - 1):
                        loss_chunk = chunk_loss / max(1, chunk_count)
                        # safety: skip non-finite chunks (prevents NaN poisoning)
                        if (not torch.isfinite(loss_chunk).all()):
                            print(f"[WARN][train] non-finite TBPTT chunk loss at step={step}, t={t}. Resetting state.")
                            optimizer.zero_grad(set_to_none=True)
                            chunk_loss = None
                            chunk_count = 0
                            inference_state = {}
                            continue
                        scaler.scale(loss_chunk).backward()

                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        # TBPTT cut: detach state tensors (memory content preserved; gradients cut)
                        inference_state = _detach_state(inference_state)

                        chunk_loss = None
                        chunk_count = 0
                else:
                    total_loss = loss_t if total_loss is None else (total_loss + loss_t)

            # keep prev track as constant (TBPTT safety)
            F_track_prev = F_track.detach()

        if not use_tbptt_chunk:
            total_loss = (total_loss / max(1, T)) if total_loss is not None else torch.tensor(0.0, device=device)
            if (not torch.isfinite(total_loss).all()):
                print(f"[WARN][train] non-finite loss at step={step}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                inference_state = {}
                continue
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

        # epoch loss uses average per-frame loss (stable across TBPTT chunking)
        avg_batch_loss = batch_loss_sum / max(1, batch_frames)
        epoch_loss += float(avg_batch_loss)

        # accumulate component losses (averaged over time) for CSV logging
        denomT = max(1, batch_frames)
        comp_sum['Total'] += float(avg_batch_loss)
        for k in comp_sum.keys():
            if k == 'Total':
                continue
            comp_sum[k] += float(batch_comp.get(k, 0.0)) / denomT
        comp_count += 1


        if step % 5 == 0:
            msg = (
                f"Batch [{step}/{len(dataloader)}] | "
                f"Tot: {avg_batch_loss:.4f} | "
                f"Seg: {(batch_comp.get('Seg_Exempt', 0.0)/max(1,batch_frames)):.4f} | "
                f"Topo: {(batch_comp.get('Topo_clDice', 0.0)/max(1,batch_frames)):.4f} | "
                f"Gate: {(batch_comp.get('KeyframeGate', 0.0)/max(1,batch_frames)):.4f}"
            )
            print(msg)

    # Return both scalar + detailed losses for logging
    denom = max(1, comp_count)
    out = {
        'loss_total': epoch_loss / max(1, len(dataloader)),
        'loss_seg': comp_sum['Seg_Exempt'] / denom,
        'loss_probe': comp_sum['Probe'] / denom,
        'loss_kin': comp_sum['Kinematic'] / denom,
        'loss_pres': comp_sum['Presence'] / denom,
        'loss_gate': comp_sum['KeyframeGate'] / denom,
    }
    return out


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, cfg, output_viz_dir=None, report_dir=None):
    """Validation/Test evaluation.

    Fixes:
    - Ensure amp_dtype is always defined.
    - Ensure M_fused_logits is always assigned (even if model call fails) -> no UnboundLocalError.
    - Compatible with 5/6 outputs.
    """
    model.eval()

    if report_dir is None:
        report_dir = getattr(cfg, "REPORT_DIR", None)

    try:
        tracker = BeyondGT_MetricsTracker(relaxation_delta=getattr(cfg, "RELAXATION_DELTA", 5))
    except TypeError:
        tracker = BeyondGT_MetricsTracker()

    use_amp = bool(getattr(cfg, "AMP", True))
    amp_dtype = _get_amp_dtype(device)
    non_blocking = bool(getattr(cfg, "NON_BLOCKING", True))
    rootness_thr = float(getattr(cfg, "ROOTNESS_THR", 0.5))

    per_frame_rows = []
    seq_rows = []

    def _bin_dt(dt):
        # PRMI protocol: first frame dt_prev=NA (dataset uses 999.0 sentinel)
        try:
            if dt is None:
                return "NA"
            if float(dt) >= 998.0:
                return "NA"
        except Exception:
            return "NA"
        if dt <= 30: return "<=30"
        if dt <= 90: return "30-90"
        return ">90"

    def _bin_seq_len(L):
        if L <= 1: return "1"
        if L <= 3: return "2-3"
        if L <= 6: return "4-6"
        return ">=7"

    def _get_meta(metas_obj, t_idx: int, b_idx: int) -> dict:
        if metas_obj is None:
            return {}
        try:
            mt = metas_obj[t_idx]
        except Exception:
            return {}
        if isinstance(mt, list):
            return mt[b_idx] if b_idx < len(mt) else {}
        if isinstance(mt, dict):
            out = {}
            for k, v in mt.items():
                try:
                    out[k] = v[b_idx]
                except Exception:
                    out[k] = v
            return out
        return {}

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device, non_blocking=non_blocking)
        masks_gt = batch["masks_gt"].cpu().numpy()
        dt_ts = batch["delta_t"].to(device, non_blocking=non_blocking)
        pad_infos = batch["pad_info"].to(device, non_blocking=non_blocking)
        metas = batch.get("meta", None)

        B, T = images.shape[:2]

        inference_state = {
            "output_dict": {}, "obj_ptr_tks": {}, "time_dict": {}, "value_dict": {},
            "prompted_output_dict": {}, "prompted_obj_ptr_tks": {}, "prompted_time_dict": {}, "prompted_value_dict": {},
            "time_days": torch.zeros(B, dtype=torch.float32, device=device),
            "prev_present": torch.zeros(B, dtype=torch.float32, device=device),
            "is_mem_empty": torch.ones(B, dtype=torch.bool, device=device),
        }

        fp_prev = [None] * B
        tepr_soil_vals = [[] for _ in range(B)]
        flush_count = [0] * B

        for t in range(T):
            img_t = images[:, t]
            dt_t = dt_ts[:, t]
            pad_t = pad_infos[:, t]

            # ✅ ensure var exists no matter what
            M_fused_logits = None

            try:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                    out = model(img_t, dt_t, inference_state, t)
                    if isinstance(out, (tuple, list)) and len(out) >= 6:
                        M_fused_logits, _, _, _, new_state = out[:5]
                    else:
                        M_fused_logits, _, _, _, new_state = out[:5]
                inference_state = new_state

                dbg = None
                if isinstance(inference_state, dict) and "__debug__" in inference_state:
                    dbg = inference_state.pop("__debug__", None)
                if dbg is not None and batch_idx == 0 and t < 2:
                    print(f"[DBG][val][t={t}] {dbg}")

            except Exception as e:
                # hard safety: skip this frame, do not crash val loop
                if batch_idx == 0 and t < 2:
                    print(f"[WARN][val] model forward failed at t={t}: {e}")
                continue

            if M_fused_logits is None:
                continue

            M_fused_1024 = F.interpolate(
                M_fused_logits.float(),
                size=(cfg.TARGET_SIZE, cfg.TARGET_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            for b in range(B):
                M_phys = PhysicalPreservingTransforms.reverse_logits_to_physical(M_fused_1024[b:b + 1], pad_t[b])
                pred_prob = torch.sigmoid(M_phys).cpu().numpy()[0, 0]
                pred_mask = (pred_prob > rootness_thr).astype(np.bool_)
                gt_mask = (masks_gt[b, t, 0] > 0.5).astype(np.bool_)
                dt_val = float(dt_t[b].detach().cpu().numpy())

                tracker.update(pred_prob, gt_mask.astype(np.float32), dt_val)

                fm = compute_frame_metrics(pred_mask, gt_mask, relaxation_delta=getattr(cfg, "RELAXATION_DELTA", 5))
                total_pixels = int(gt_mask.size)
                flush_flag = False
                try:
                    if isinstance(inference_state, dict) and ("is_mem_empty" in inference_state) and inference_state["is_mem_empty"] is not None:
                        v = inference_state["is_mem_empty"]
                        if torch.is_tensor(v):
                            flush_flag = bool(v[b].detach().cpu().item())
                        else:
                            flush_flag = bool(v)
                    else:
                        # fallback: approximate empty by checking bank dict sizes
                        n_recent = len(inference_state.get("output_dict", {})) if isinstance(inference_state, dict) else 0
                        n_prompt = len(inference_state.get("prompted_output_dict", {})) if isinstance(inference_state, dict) else 0
                        flush_flag = (n_recent + n_prompt) == 0
                except Exception:
                    flush_flag = False
                flush_count[b] += int(flush_flag)

                tepr_delta_fp_pct = ""
                if fp_prev[b] is not None:
                    d_fp = max(0.0, float(fm["fp"]) - float(fp_prev[b]))
                    tepr_delta_fp_pct = 100.0 * d_fp / max(1, total_pixels)
                    if fm["gt_pixels"] == 0:
                        tepr_soil_vals[b].append(float(tepr_delta_fp_pct))
                fp_prev[b] = float(fm["fp"])

                meta_bt = _get_meta(metas, t, b)
                seq_len = int(meta_bt.get("seq_len", T))
                flips = int(meta_bt.get("flips", 0))
                mixed = bool(meta_bt.get("mixed", flips >= 1))

                per_frame_rows.append({
                    "subset": meta_bt.get("subset", ""),
                    "split": meta_bt.get("split", ""),
                    "seq_id": meta_bt.get("seq_id", ""),
                    "seq_len": seq_len,
                    "seq_len_bin": _bin_seq_len(seq_len),
                    "flips": flips,
                    "mixed": mixed,
                    "frame_idx": int(meta_bt.get("frame_idx", t)),
                    "delta_t": dt_val,
                    "dt_bin": _bin_dt(dt_val),
                    "has_root": int(meta_bt.get("has_root", int(fm["gt_pixels"] > 0))),
                    "flush": int(flush_flag),
                    "image_name": meta_bt.get("image_name", ""),
                    "timestamp": meta_bt.get("timestamp", ""),
                    "total_pixels": total_pixels,
                    "tp": int(fm["tp"]),
                    "fp": int(fm["fp"]),
                    "fn": int(fm["fn"]),
                    "tn": int(fm["tn"]),
                    "pred_pixels": int(fm["pred_pixels"]),
                    "gt_pixels": int(fm["gt_pixels"]),
                    "iou": float(fm["iou"]),
                    "dice": float(fm["dice"]),
                    "precision": float(fm["precision"]),
                    "recall": float(fm["recall"]),
                    "cldice": float(fm["cldice"]),
                    "sdf_relaxed_iou": float(fm["sdf_relaxed_iou"]),
                    "tepr_delta_fp_pct": tepr_delta_fp_pct,
                })

        # sequence summary
        for b in range(B):
            meta0 = _get_meta(metas, 0, b)
            seq_id = meta0.get("seq_id", f"seq_{batch_idx}_{b}")
            seq_rows.append({
                "subset": meta0.get("subset", ""),
                "split": meta0.get("split", ""),
                "seq_id": seq_id,
                "seq_len": int(meta0.get("seq_len", T)),
                "flips": int(meta0.get("flips", 0)),
                "mixed": bool(meta0.get("mixed", False)),
                "flush_count": int(flush_count[b]),
                "tepr_soil_mean": float(np.mean(tepr_soil_vals[b])) if len(tepr_soil_vals[b]) > 0 else "",
                "tepr_soil_max": float(np.max(tepr_soil_vals[b])) if len(tepr_soil_vals[b]) > 0 else "",
            })

    res = tracker.summarize() if hasattr(tracker, "summarize") else (tracker.compute_summary() if hasattr(tracker, "compute_summary") else {})

    # write reports
    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        per_frame_path = os.path.join(report_dir, "per_frame_metrics.csv")
        seq_path = os.path.join(report_dir, "sequence_summary.csv")

        if len(per_frame_rows) > 0:
            fieldnames = list(per_frame_rows[0].keys())
            with open(per_frame_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in per_frame_rows:
                    w.writerow(r)

        if len(seq_rows) > 0:
            fieldnames = list(seq_rows[0].keys())
            with open(seq_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in seq_rows:
                    w.writerow(r)

        try:
            from tools.stratified_report import build_stratified_report
            build_stratified_report(per_frame_path, report_dir)
        except Exception as e:
            print(f"⚠️ Stratified report generation skipped due to error: {e}")

    return res
