"""Dual FIFO memory state manager (recent + prompted).

This file exists to keep `models/root_sam_pro.py` as an *orchestrator*.

Key properties:
  - Functional: never mutates the input state dict.
  - Batch-aware flush: per-sample reset masks history without resurrecting artifacts.
  - Priority eviction: uses per-frame `value_dict` if available, else oldest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch

from .memory_router import mean_value


@dataclass
class DualBankState:
    # recent
    out_dict: Dict[int, torch.Tensor]
    obj_ptrs: Dict[int, torch.Tensor]
    time_dict: Dict[int, torch.Tensor]
    value_dict: Dict[int, torch.Tensor]
    # prompted
    p_out_dict: Dict[int, torch.Tensor]
    p_obj_ptrs: Dict[int, torch.Tensor]
    p_time_dict: Dict[int, torch.Tensor]
    p_value_dict: Dict[int, torch.Tensor]
    # scalars
    time_days: torch.Tensor          # (B,)
    prev_present: torch.Tensor       # (B,)
    is_mem_empty: torch.Tensor       # (B,) bool


class DualMemoryBank:
    """Read/update helper around the checkpoint-safe dict state."""

    @staticmethod
    def read(state_in: Any, *, B: int, device: torch.device, frame_idx: int) -> DualBankState:
        if (not isinstance(state_in, dict)) or frame_idx == 0:
            return DualBankState(
                out_dict={}, obj_ptrs={}, time_dict={}, value_dict={},
                p_out_dict={}, p_obj_ptrs={}, p_time_dict={}, p_value_dict={},
                time_days=torch.zeros(B, dtype=torch.float32, device=device),
                prev_present=torch.zeros(B, dtype=torch.float32, device=device),
                is_mem_empty=torch.ones(B, dtype=torch.bool, device=device),
            )

        def _d(key):
            v = state_in.get(key, {})
            return dict(v) if isinstance(v, dict) else {}

        time_days = state_in.get("time_days", torch.zeros(B, dtype=torch.float32, device=device))
        is_empty = state_in.get("is_mem_empty", torch.ones(B, dtype=torch.bool, device=device))
        prev_present = state_in.get("prev_present", torch.zeros(B, dtype=torch.float32, device=device))

        return DualBankState(
            out_dict=_d("output_dict"),
            obj_ptrs=_d("obj_ptr_tks"),
            time_dict=_d("time_dict"),
            value_dict=_d("value_dict"),
            p_out_dict=_d("prompted_output_dict"),
            p_obj_ptrs=_d("prompted_obj_ptr_tks"),
            p_time_dict=_d("prompted_time_dict"),
            p_value_dict=_d("prompted_value_dict"),
            time_days=time_days,
            prev_present=prev_present,
            is_mem_empty=is_empty,
        )

    @staticmethod
    def _mask_dict_bchw(d: Dict[int, torch.Tensor], keep_mask4: torch.Tensor):
        for k in list(d.keys()):
            v = d.get(k, None)
            if torch.is_tensor(v) and v.dim() >= 4 and v.shape[0] == keep_mask4.shape[0]:
                d[k] = v * keep_mask4

    @staticmethod
    def _mask_ptr_dict(d: Dict[int, torch.Tensor], keep_mask2: torch.Tensor, reset_flags: torch.Tensor, target_dtype: torch.dtype):
        B = int(reset_flags.shape[0])
        for k in list(d.keys()):
            v = d.get(k, None)
            if not torch.is_tensor(v) or v.shape[0] != B:
                continue
            if v.dim() == 2:
                d[k] = v * keep_mask2
            elif v.dim() == 3:
                d[k] = v * (~reset_flags).view(B, 1, 1).to(dtype=target_dtype, device=v.device)
            else:
                shape = [B] + [1] * (v.dim() - 1)
                d[k] = v * (~reset_flags).view(*shape).to(dtype=target_dtype, device=v.device)

    @staticmethod
    def _mask_value_dict(d: Dict[int, torch.Tensor], reset_flags: torch.Tensor):
        B = int(reset_flags.shape[0])
        for k in list(d.keys()):
            v = d.get(k, None)
            if torch.is_tensor(v) and v.shape[0] == B:
                d[k] = v * (~reset_flags).to(dtype=v.dtype, device=v.device)

    @staticmethod
    def _prune_fifo(
        out_dict: Dict[int, torch.Tensor],
        obj_ptrs: Dict[int, torch.Tensor],
        time_dict: Dict[int, torch.Tensor],
        value_dict: Dict[int, torch.Tensor],
        max_frames: int,
    ):
        if max_frames <= 0:
            return
        if len(out_dict) <= max_frames:
            return
        keys = list(out_dict.keys())
        if isinstance(value_dict, dict) and len(value_dict) > 0:
            k_pop = min(keys, key=lambda kk: mean_value(value_dict.get(kk, None)))
        else:
            k_pop = min(keys)
        out_dict.pop(k_pop, None)
        obj_ptrs.pop(k_pop, None)
        time_dict.pop(k_pop, None)
        value_dict.pop(k_pop, None)

    @staticmethod
    def write_new_state(
        *,
        old: DualBankState,
        frame_idx: int,
        mem_feat_bchw: torch.Tensor,
        obj_ptr_decoder: torch.Tensor | None,
        curr_time_days: torch.Tensor,
        write_gate: torch.Tensor,
        split_weight: torch.Tensor,
        val_prob2: torch.Tensor,
        p_present: torch.Tensor,
        reset_flags: torch.Tensor,
        new_is_mem_empty: torch.Tensor,
        max_recent_frames: int,
        max_prompted_frames: int,
        target_dtype: torch.dtype,
        debug_events: Any = None,
    ) -> dict:
        """Functional update: (old_state, decisions) -> new_state."""

        B = int(mem_feat_bchw.shape[0])
        device = mem_feat_bchw.device

        # clone dicts
        new_out = {k: v for k, v in old.out_dict.items()}
        new_ptr = {k: v for k, v in old.obj_ptrs.items()}
        new_time = {k: v for k, v in old.time_dict.items()}
        new_val = {k: v for k, v in old.value_dict.items()} if isinstance(old.value_dict, dict) else {}

        new_p_out = {k: v for k, v in old.p_out_dict.items()}
        new_p_ptr = {k: v for k, v in old.p_obj_ptrs.items()}
        new_p_time = {k: v for k, v in old.p_time_dict.items()}
        new_p_val = {k: v for k, v in old.p_value_dict.items()} if isinstance(old.p_value_dict, dict) else {}

        # flush/reset masks
        if reset_flags is not None and reset_flags.any():
            keep_mask4 = (~reset_flags).view(B, 1, 1, 1).to(dtype=target_dtype, device=device)
            keep_mask2 = (~reset_flags).view(B, 1).to(dtype=target_dtype, device=device)

            for d in (new_out, new_p_out):
                DualMemoryBank._mask_dict_bchw(d, keep_mask4)

            for d in (new_ptr, new_p_ptr):
                DualMemoryBank._mask_ptr_dict(d, keep_mask2, reset_flags, target_dtype)

            for d in (new_val, new_p_val):
                DualMemoryBank._mask_value_dict(d, reset_flags)

            if bool(reset_flags.all().item()):
                new_out, new_ptr, new_time, new_val = {}, {}, {}, {}
                new_p_out, new_p_ptr, new_p_time, new_p_val = {}, {}, {}, {}

        # SoftWrite: split into banks
        prompt_w = (write_gate * split_weight).view(B, 1, 1, 1).to(target_dtype)
        recent_w = (write_gate * (1.0 - split_weight)).view(B, 1, 1, 1).to(target_dtype)

        if (recent_w > 1e-6).any():
            new_out[int(frame_idx)] = mem_feat_bchw * recent_w
            new_time[int(frame_idx)] = curr_time_days
            if obj_ptr_decoder is not None:
                new_ptr[int(frame_idx)] = obj_ptr_decoder * recent_w.view(B, 1, 1).to(target_dtype)
            val_recent = (recent_w.view(B).detach().float() * val_prob2[:, 0].detach().float() * p_present.detach().float())
            new_val[int(frame_idx)] = val_recent

        if (prompt_w > 1e-6).any():
            new_p_out[int(frame_idx)] = mem_feat_bchw * prompt_w
            new_p_time[int(frame_idx)] = curr_time_days
            if obj_ptr_decoder is not None:
                new_p_ptr[int(frame_idx)] = obj_ptr_decoder * prompt_w.view(B, 1, 1).to(target_dtype)
            val_prompt = (prompt_w.view(B).detach().float() * val_prob2[:, 1].detach().float() * p_present.detach().float())
            new_p_val[int(frame_idx)] = val_prompt

        # FIFO prune (priority by value)
        DualMemoryBank._prune_fifo(new_out, new_ptr, new_time, new_val, int(max_recent_frames))
        DualMemoryBank._prune_fifo(new_p_out, new_p_ptr, new_p_time, new_p_val, int(max_prompted_frames))

        new_state = {
            "output_dict": new_out,
            "obj_ptr_tks": new_ptr,
            "time_dict": new_time,
            "value_dict": new_val,
            "prompted_output_dict": new_p_out,
            "prompted_obj_ptr_tks": new_p_ptr,
            "prompted_time_dict": new_p_time,
            "prompted_value_dict": new_p_val,
            "time_days": curr_time_days,
            "prev_present": p_present.detach(),
            "is_mem_empty": new_is_mem_empty,
        }
        if debug_events is not None:
            new_state["__debug__"] = debug_events
        return new_state
