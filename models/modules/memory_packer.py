import math
import torch
import torch.nn.functional as F

from .memory_router import mean_value


def pack_memory_bank(
    host,
    out_dict: dict,
    obj_ptr_dict: dict,
    time_dict: dict,
    out_value: dict | None,
    prompted_out: dict,
    prompted_ptr: dict,
    prompted_time: dict,
    prompted_value: dict | None,
    target_hw,
    curr_time_days: torch.Tensor,
):
    """Pack dual memory banks into a single token sequence for SAM2 memory attention.

    This is intentionally implemented as a standalone function for:
      - reproducibility (pure input->output)
      - testability (can be unit-tested with a stub host)
      - keeping RootSAMPro as an orchestrator

    Dependencies expected on `host`:
      - attributes: mem_store_dim, ptr_dim, attn_kv_in_dim, max_recent_frames, max_prompted_frames, tbptt_keep_last,
                   value_half_life_raw, mem_pool_stride, use_temporal_pos, use_bio_kes, add_tpos_to_ptrs,
                   bio_half_life_raw, ptr_half_life_raw, srd_mem_proj
      - methods: _pool_mem_map(v_bchw)->v_bchw, _to_tokens_bLc(x_bchw)->(B,L,C), _project_ptr_to_kv(ptr)->(B,K,kv)
    """
    has_recent = isinstance(out_dict, dict) and len(out_dict) > 0
    has_prompted = isinstance(prompted_out, dict) and len(prompted_out) > 0
    if (not has_recent) and (not has_prompted):
        return None, 0

    def _score_value(val_tensor, t_tensor, hl_days: float) -> float:
        base = mean_value(val_tensor)
        if (t_tensor is None) or (not torch.is_tensor(t_tensor)) or (hl_days <= 1e-6):
            return base
        try:
            age = (curr_time_days.float() - t_tensor.float()).clamp(min=0.0)
            age_mean = float(age.detach().float().mean().item())
            decay = math.exp(-math.log(2.0) * age_mean / hl_days)
            return base * decay
        except Exception:
            return base

    hl_days = float(F.softplus(host.value_half_life_raw.detach().float()).clamp(min=1e-3).cpu().item())

    def _select_keys(bank_dict: dict, val_dict: dict | None, t_dict: dict, cap: int, must_keep: set[int]):
        if not isinstance(bank_dict, dict) or len(bank_dict) == 0 or cap <= 0:
            return []
        keys_all = sorted(list(bank_dict.keys()))
        # always keep TBPTT tail keys if they exist in this bank
        keep = [k for k in keys_all if k in must_keep]
        keep_set = set(keep)

        if val_dict is None or (not isinstance(val_dict, dict)) or len(val_dict) == 0:
            # fallback: most recent
            keys = keys_all[-cap:]
            # ensure must_keep included
            for k in keep:
                if k not in keys:
                    keys.append(k)
            uniq = sorted(list(set(keys)))
            return uniq[-cap:]

        # score by (time-decayed) learned value (higher is better)
        scored = []
        for k in keys_all:
            if k in keep_set:
                continue
            v = val_dict.get(k, None)
            t = t_dict.get(k, None) if isinstance(t_dict, dict) else None
            scored.append((_score_value(v, t, hl_days), k))
        scored.sort(key=lambda x: x[0], reverse=True)

        keys = keep[:]
        for _, k in scored:
            if len(keys) >= cap:
                break
            keys.append(k)
        # stable ordering by frame index for reproducibility
        return sorted(list(set(keys)))

    keep_k = max(0, int(host.tbptt_keep_last))
    # must-keep keys are defined over the union of *all* existing frames (paper: TBPTT tail is never detached)
    union_keys = sorted(list(set(list(out_dict.keys()) + list(prompted_out.keys()))))
    must_keep = set(union_keys[-keep_k:]) if keep_k > 0 else set()
    grad_keys = must_keep

    recent_keys = _select_keys(out_dict, out_value, time_dict, int(host.max_recent_frames), must_keep) if has_recent else []
    prompted_keys = _select_keys(prompted_out, prompted_value, prompted_time, int(host.max_prompted_frames), must_keep) if has_prompted else []

    mem_spatial = []
    mem_ptrs = []

    def _append_frame(v_map_raw, v_ptr_raw, t_mem, temporal_on: bool, detach_old: bool):
        # ---- spatial map ----
        if v_map_raw is not None and torch.is_tensor(v_map_raw) and v_map_raw.dim() == 4:
            v_map = None
            # BCHW
            if int(v_map_raw.shape[1]) == int(host.mem_store_dim):
                v_map = v_map_raw.contiguous()
            # BHWC
            elif int(v_map_raw.shape[-1]) == int(host.mem_store_dim):
                v_map = v_map_raw.permute(0, 3, 1, 2).contiguous()

            if v_map is not None:
                if v_map.shape[-2:] != tuple(target_hw):
                    v_map = F.interpolate(v_map, size=target_hw, mode="bilinear", align_corners=False)
                if detach_old:
                    v_map = v_map.detach()

                v_map_p = host._pool_mem_map(v_map)
                tok = host._to_tokens_bLc(v_map_p)  # (B,L,store_dim)
                tok = host.srd_mem_proj(tok)        # -> (B,L,kv_in_dim)

                if temporal_on and host.use_temporal_pos and (t_mem is not None):
                    age = (curr_time_days.float() - t_mem.float()).clamp(min=0.0)
                    if host.use_bio_kes:
                        # decay(age) = exp(-ln2 * age / half_life)
                        hl = F.softplus(host.bio_half_life_raw.float()).clamp(min=1e-3)  # (1,1,C)
                        expo = -(math.log(2.0) * age.view(v_map.shape[0], 1, 1) / hl).clamp(min=-60.0, max=0.0)
                        tok = tok * torch.exp(expo).to(tok.dtype)

                mem_spatial.append(tok)

        # ---- obj ptrs (raw stored as 256; project at pack time) ----
        if v_ptr_raw is not None and torch.is_tensor(v_ptr_raw):
            v_ptr = v_ptr_raw
            if v_ptr.dim() == 2:
                v_ptr = v_ptr.unsqueeze(1)

            if int(v_ptr.shape[-1]) == int(host.ptr_dim):
                v_ptr_kv = host._project_ptr_to_kv(v_ptr)
            elif int(v_ptr.shape[-1]) == int(host.attn_kv_in_dim):
                v_ptr_kv = v_ptr
            else:
                v_ptr_kv = None

            if v_ptr_kv is not None and int(v_ptr_kv.shape[-1]) == int(host.attn_kv_in_dim):
                if detach_old:
                    v_ptr_kv = v_ptr_kv.detach()

                if temporal_on and host.use_temporal_pos and host.add_tpos_to_ptrs and (t_mem is not None):
                    age = (curr_time_days.float() - t_mem.float()).clamp(min=0.0)
                    if host.use_bio_kes:
                        hl = F.softplus(host.ptr_half_life_raw.float()).clamp(min=1e-3)  # (1,1,C)
                        expo = -(math.log(2.0) * age.view(v_ptr_kv.shape[0], 1, 1) / hl).clamp(min=-60.0, max=0.0)
                        v_ptr_kv = v_ptr_kv * torch.exp(expo).to(v_ptr_kv.dtype)

                mem_ptrs.append(v_ptr_kv.contiguous())

    # prompted first (no temporal bias by default)
    for k in prompted_keys:
        _append_frame(
            prompted_out.get(k),
            prompted_ptr.get(k) if isinstance(prompted_ptr, dict) else None,
            prompted_time.get(k) if isinstance(prompted_time, dict) else None,
            False,
            k not in grad_keys,
        )

    # recent bank (temporal bias on)
    for k in recent_keys:
        _append_frame(
            out_dict.get(k),
            obj_ptr_dict.get(k) if isinstance(obj_ptr_dict, dict) else None,
            time_dict.get(k) if isinstance(time_dict, dict) else None,
            True,
            k not in grad_keys,
        )

    if len(mem_spatial) == 0:
        return None, 0

    mem_tokens_b = torch.cat(mem_spatial, dim=1)  # (B,L,kv_in)
    num_obj_ptrs = 0
    if len(mem_ptrs) > 0:
        ptr_tokens = torch.cat(mem_ptrs, dim=1)   # (B,K',kv_in)
        num_obj_ptrs = int(ptr_tokens.shape[1])
        mem_tokens_b = torch.cat([ptr_tokens, mem_tokens_b], dim=1)

    return mem_tokens_b, num_obj_ptrs
