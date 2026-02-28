"""SAM2.1 compatibility adapter.

RootSAM-Pro intentionally does **not** vendor Meta SAM2.1 code.
However, Meta's `build_sam2` and `build_sam2_video_predictor` objects
can differ slightly across versions / checkpoints.

This adapter:
  - Builds the SAM2 object (image model or video predictor fallback)
  - Provides *stable* wrappers for:
      forward_image / mask_decoder / prompt_encoder / memory_attention / memory_encoder
  - Adds safe fallbacks when an API key is missing.

Goal: make the RootSAM-Pro model code resilient (top-journal reproducibility).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch


class Sam2Compat:
    def __init__(self, sam2: Any):
        self.sam2 = sam2

    # -----------------------------
    # Build helpers
    # -----------------------------
    @staticmethod
    def build(model_cfg: str, checkpoint: str):
        """Try build_sam2 first; fall back to build_sam2_video_predictor."""
        try:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
        except Exception as e:
            raise ImportError(
                "SAM2 is not available. Please clone Meta SAM2 into ./sam2 and pip install -e ."
            ) from e

        try:
            sam2 = build_sam2(model_cfg, checkpoint)
        except Exception:
            sam2 = build_sam2_video_predictor(model_cfg, checkpoint)
        return Sam2Compat(sam2)

    # -----------------------------
    # Stable wrappers
    # -----------------------------
    def forward_image(self, x: torch.Tensor) -> Dict[str, Any]:
        if hasattr(self.sam2, "forward_image"):
            out = self.sam2.forward_image(x)
            return out if isinstance(out, dict) else {"vision_features": out}

        # Fallback: try image_encoder
        if hasattr(self.sam2, "image_encoder"):
            feats = self.sam2.image_encoder(x)
            return {"vision_features": feats}
        raise AttributeError("SAM2 object has neither forward_image nor image_encoder")

    def prompt_encode_empty(self, B: int, device: torch.device, dtype: torch.dtype):
        """Return (sparse_prompt_embeddings, dense_prompt_embeddings) for 'no prompt'."""
        if not hasattr(self.sam2, "prompt_encoder"):
            # safe fallback: dense PE zeros
            return None, None

        empty_pts = torch.empty((B, 0, 2), device=device, dtype=torch.float32)
        empty_lbls = torch.empty((B, 0), device=device, dtype=torch.int32)
        with torch.no_grad():
            sparse_pe, _ = self.sam2.prompt_encoder(points=(empty_pts, empty_lbls), boxes=None, masks=None)

        no_mask_embed = self.sam2.prompt_encoder.no_mask_embed.weight
        # dense prompt embedding is 1xCx1x1 expanded later by caller
        dense_base = no_mask_embed.reshape(1, -1, 1, 1).to(device=device, dtype=dtype)
        return sparse_pe.to(device=device, dtype=dtype), dense_base

    def decode_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_features=None,
    ):
        if not hasattr(self.sam2, "mask_decoder"):
            raise AttributeError("SAM2 object missing mask_decoder")
        return self.sam2.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
            high_res_features=high_res_features,
        )

    def memory_attention(self, curr_features, curr_pos, memory_dict, obj_ptr_tks):
        if not hasattr(self.sam2, "memory_attention"):
            raise AttributeError("SAM2 object missing memory_attention")

        fn = self.sam2.memory_attention
        sig = None
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        # Try common call patterns
        if sig and "memory_dict" in sig.parameters:
            return fn(curr_features=curr_features, curr_pos=curr_pos, memory_dict=memory_dict, obj_ptr_tks=obj_ptr_tks)
        if sig and "memories" in sig.parameters:
            return fn(curr_features=curr_features, curr_pos=curr_pos, memories=memory_dict, obj_ptr_tks=obj_ptr_tks)

        # Fallback to positional call
        return fn(curr_features, curr_pos, memory_dict, obj_ptr_tks)

    def memory_encode(self, image_t, mask_prob, pix_feat=None):
        if not hasattr(self.sam2, "memory_encoder"):
            raise AttributeError("SAM2 object missing memory_encoder")
        try:
            return self.sam2.memory_encoder(image_t, mask_prob)
        except TypeError:
            return self.sam2.memory_encoder(pix_feat=pix_feat, masks=mask_prob)
