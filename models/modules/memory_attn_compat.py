"""SAM2 Memory Attention compatibility wrapper.

This wrapper is responsible for:
  - packing dual-bank memories into KV tokens
  - formatting query/memory token sequences
  - adapting call signatures across SAM2 variants
  - providing robust fallbacks (never crash training)

Patch vP0~P2:
  ✅ Add *spatial* 2D sin-cos positional encodings (query dim) for both query and spatial memory tokens.
     Pointer tokens do not have 2D location -> zeros.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from .dual_memory_bank import DualBankState
from .memory_packer import pack_memory_bank


def _build_2d_sincos_pos(H: int, W: int, dim: int, *, device, dtype) -> torch.Tensor:
    """Return (H*W, dim) 2D sin-cos positional encoding.

    Notes:
      - dim must be divisible by 4 for a standard 2D (y,x) split.
      - We compute in fp32 for stability then cast to `dtype`.
    """
    if H <= 0 or W <= 0 or dim <= 0:
        return torch.zeros((max(0, H * W), max(0, dim)), device=device, dtype=dtype)
    if dim % 4 != 0:
        return torch.zeros((H * W, dim), device=device, dtype=dtype)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    half = dim // 2
    quarter = half // 2
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(1.0, float(quarter))))

    out_y = yy[:, None] * omega[None, :]
    out_x = xx[:, None] * omega[None, :]
    pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    pe = torch.cat([pe_y, pe_x], dim=1)
    return pe.to(dtype=dtype)


def memory_attention_compat(
    host,
    *,
    F_base: torch.Tensor,
    bank_old: DualBankState,
    curr_time_days: torch.Tensor,
    target_hw: Tuple[int, int],
    debug_events: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Run SAM2 memory attention in a KV-dim-safe, signature-safe manner.

    Returns:
      F_track_raw: (B, attn_d_model, H, W)
      sim: (B,) cosine similarity between tracked and base features
      debug_events: passthrough list (may be appended)

    Safety:
      - Never raises (falls back to single-frame features)
    """

    B = int(F_base.shape[0])
    device = F_base.device

    # Quick bypass if globally empty or no stored frames
    if bank_old.is_mem_empty.all() or ((len(bank_old.out_dict) == 0) and (len(bank_old.p_out_dict) == 0)):
        return F_base, torch.ones(B, device=device), debug_events

    Hf, Wf = int(target_hw[0]), int(target_hw[1])

    # Pack dual memory into a single token sequence (B,Lm,kv_in_dim)
    mem_b, num_obj_ptrs = pack_memory_bank(
        host,
        bank_old.out_dict,
        bank_old.obj_ptrs,
        bank_old.time_dict,
        bank_old.value_dict,
        bank_old.p_out_dict,
        bank_old.p_obj_ptrs,
        bank_old.p_time_dict,
        bank_old.p_value_dict,
        (Hf, Wf),
        curr_time_days,
    )
    if mem_b is None:
        return F_base, torch.ones(B, device=device), debug_events

    # Query tokens (Lq,B,Cq)
    curr_b = host._to_tokens_bLc(F_base)
    curr_seq = curr_b.transpose(0, 1).contiguous()

    # Memory tokens (Lm,B,Ckv)
    mem_seq = mem_b.transpose(0, 1).contiguous()

    # ------------------------------------------------------------
    # Positional encoding (query dim)
    # ------------------------------------------------------------
    use_spatial_pos = bool(getattr(getattr(host, "cfg", None), "USE_SPATIAL_POS", True))
    if not use_spatial_pos:
        curr_pos_seq = curr_seq.new_zeros(curr_seq.shape)
        mem_pos_seq = curr_seq.new_zeros((mem_seq.shape[0], curr_seq.shape[1], curr_seq.shape[2]))
    else:
        # Query PE: (Lq, Cq) -> (Lq,B,Cq)
        pe_q = _build_2d_sincos_pos(Hf, Wf, int(host.attn_d_model), device=curr_seq.device, dtype=torch.float32)
        pe_q = pe_q.to(dtype=curr_seq.dtype)
        curr_pos_seq = pe_q.unsqueeze(1).expand(-1, curr_seq.shape[1], -1).contiguous()

        # Memory PE: ptr tokens first -> zeros; spatial tokens -> repeated 2D PE of pooled grid.
        num_ptr = int(num_obj_ptrs)
        Lm = int(mem_seq.shape[0])
        Lsp = max(0, Lm - num_ptr)
        s = int(getattr(host, "mem_pool_stride", 1))
        Hm = max(1, Hf // max(1, s))
        Wm = max(1, Wf // max(1, s))
        L_frame = int(Hm * Wm)

        if Lsp > 0 and L_frame > 0 and (Lsp % L_frame == 0):
            n_frames = int(Lsp // L_frame)
            pe_m_frame = _build_2d_sincos_pos(Hm, Wm, int(host.attn_d_model), device=curr_seq.device, dtype=torch.float32)
            pe_m_frame = pe_m_frame.to(dtype=curr_seq.dtype)
            pe_m_spatial = pe_m_frame.repeat(n_frames, 1)
        else:
            pe_m_spatial = curr_seq.new_zeros((Lsp, int(host.attn_d_model)))

        pe_ptr = curr_seq.new_zeros((num_ptr, int(host.attn_d_model)))
        pe_m = torch.cat([pe_ptr, pe_m_spatial], dim=0)
        if pe_m.shape[0] != Lm:
            pe_m = curr_seq.new_zeros((Lm, int(host.attn_d_model)))

        mem_pos_seq = pe_m.unsqueeze(1).expand(-1, curr_seq.shape[1], -1).contiguous()

    try:
        # SAM2 variants: some take `num_obj_ptrs` arg, some don't.
        try:
            out_seq = host.sam2.memory_attention(curr_seq, mem_seq, curr_pos_seq, mem_pos_seq, int(num_obj_ptrs))
        except TypeError:
            out_seq = host.sam2.memory_attention(curr_seq, mem_seq, curr_pos_seq, mem_pos_seq)

        out_seq = out_seq[0] if isinstance(out_seq, (tuple, list)) else out_seq
        if not (torch.is_tensor(out_seq) and out_seq.dim() == 3):
            raise RuntimeError("memory_attention returned non-tensor or wrong rank")

        # (Lq,B,Cq) -> (B,Cq,H,W)
        F_track_raw = (
            out_seq.transpose(0, 1).contiguous().transpose(1, 2).reshape(B, int(host.attn_d_model), Hf, Wf).contiguous()
        )
        sim = F.cosine_similarity(F_track_raw.flatten(1), F_base.flatten(1), dim=1).clamp(0.0, 1.0)
        return F_track_raw, sim, debug_events

    except Exception as e:
        if debug_events is not None:
            try:
                debug_events.append({"tag": "mem_attn", "action": "runtime_exception", "error": str(e)})
            except Exception:
                pass
        return F_base, torch.ones(B, device=device), debug_events
