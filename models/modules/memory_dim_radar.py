"""Dimension radar utilities for SAM2 compatibility.

This module centralizes best-effort detection of key dimensions and official
projection availability, so that `models/root_sam_pro.py` stays an orchestrator.

We intentionally keep this conservative:
  - If detection fails, we fall back to known-good defaults (e.g., SAM2.1 Hiera-L).
  - No side effects: only reads from the SAM2 object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MemoryDims:
    """Detected / configured dimensions for memory attention plumbing."""
    attn_d_model: int = 256          # query / image token dim
    mem_store_dim: int = 64          # memory encoder output channels (spatial memory map)
    ptr_dim: int = 256               # object pointer token dim (bookkeeping / similarity)
    kv_in_dim: int = 64              # KV input dim expected by memory_attention
    kv_in_dim_detected: Optional[int] = None
    mem_store_dim_detected: Optional[int] = None
    ptr_dim_detected: Optional[int] = None


def detect_kv_in_dim(sam2) -> Optional[int]:
    """Best-effort detect memory_attention kv_in_dim from SAM2 modules."""
    try:
        ma = getattr(sam2, "memory_attention", None)
        if ma is None:
            return None
        layers = getattr(ma, "layers", None)
        if not layers:
            return None
        layer0 = layers[0]
        ca = getattr(layer0, "cross_attn_image", None)
        if ca is None:
            return None
        kp = getattr(ca, "k_proj", None)
        if kp is None:
            return None
        if hasattr(kp, "in_features"):
            v = int(kp.in_features)
            return v if v > 0 else None
        return None
    except Exception:
        return None


def detect_mem_store_dim(sam2) -> Optional[int]:
    """Best-effort detect memory encoder store dim (out channels)."""
    try:
        me = getattr(sam2, "memory_encoder", None)
        if me is None:
            return None
        # common: memory_encoder.out_dim
        if hasattr(me, "out_dim"):
            v = int(getattr(me, "out_dim"))
            return v if v > 0 else None
        # sometimes: proj / output conv
        for attr in ["proj", "output_proj", "out_proj", "conv_out"]:
            m = getattr(me, attr, None)
            if m is None:
                continue
            if hasattr(m, "out_channels"):
                v = int(m.out_channels)
                return v if v > 0 else None
        return None
    except Exception:
        return None


def detect_ptr_dim(sam2) -> Optional[int]:
    """Best-effort detect pointer token dim (input dim of official obj_ptr_proj if present)."""
    try:
        proj = get_official_obj_ptr_proj(sam2)
        if proj is None:
            return None
        if hasattr(proj, "in_features"):
            v = int(proj.in_features)
            return v if v > 0 else None
        return None
    except Exception:
        return None


def has_official_obj_ptr_proj(sam2) -> bool:
    """Whether SAM2 provides a pretrained object pointer projection."""
    return get_official_obj_ptr_proj(sam2) is not None


def get_official_obj_ptr_proj(sam2):
    """Return the official obj_ptr_proj module if available, else None."""
    if hasattr(sam2, "obj_ptr_proj"):
        return getattr(sam2, "obj_ptr_proj")
    md = getattr(sam2, "mask_decoder", None)
    if md is not None and hasattr(md, "obj_ptr_proj"):
        return getattr(md, "obj_ptr_proj")
    return None


def detect_memory_dims(
    sam2,
    *,
    attn_d_model_default: int = 256,
    mem_store_dim_default: int = 64,
    ptr_dim_default: int = 256,
    kv_in_dim_default: int = 64,
) -> MemoryDims:
    """Detect memory-related dims with safe fallbacks."""
    kv = detect_kv_in_dim(sam2)
    ms = detect_mem_store_dim(sam2)
    pd = detect_ptr_dim(sam2)

    kv_final = kv_in_dim_default if not isinstance(kv, int) or kv <= 0 else int(kv)
    ms_final = mem_store_dim_default if not isinstance(ms, int) or ms <= 0 else int(ms)
    pd_final = ptr_dim_default if not isinstance(pd, int) or pd <= 0 else int(pd)

    return MemoryDims(
        attn_d_model=int(attn_d_model_default),
        mem_store_dim=int(ms_final),
        ptr_dim=int(pd_final),
        kv_in_dim=int(kv_final),
        kv_in_dim_detected=kv,
        mem_store_dim_detected=ms,
        ptr_dim_detected=pd,
    )
