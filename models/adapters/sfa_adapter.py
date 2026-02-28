import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter_utils import VanillaAdapter, SRDSafeWrapper
from .srd_adapter import SRDAdapter
from ..modules import BHFI


class SFAAdapter(nn.Module):
    """SFA: Spatial-Frequency Adapter (SRD + BHFI + HR soil washers).

    Responsibilities:
      1) Inject SRD/Vanilla adapters into SAM2 image encoder FFN (block.mlp).
      2) Provide optional HR-FPN soil washers and BHFI refinement for mask decoding.
      3) Optionally enhance memory store maps via SRD on mem_store_dim tokens.

    NOTE:
      - SRD injection happens by wrapping SAM2 blocks in-place, but we keep
        the adapter modules registered here so they are trainable & checkpointed.
    """

    def __init__(self, cfg, mode: str, sam2, attn_d_model: int, mem_store_dim: int):
        super().__init__()
        self.cfg = cfg
        self.mode = str(mode)
        self.attn_d_model = int(attn_d_model)
        self.mem_store_dim = int(mem_store_dim)

        # Whether SFA is considered "on" for the paper (SRD + BHFI present).
        self.enabled = self.mode in ["SFA_ONLY", "SFA_ASTA", "FULL", "SRD_ONLY", "SRD_KMR"]

        # --- inject adapter blocks into image encoder ---
        self.srd_img_blocks = nn.ModuleList()
        self._inject_into_image_encoder(sam2)

        # IMPORTANT:
        # SRD adapters are injected into SAM2 blocks (as submodules). Some pipelines
        # may freeze SAM2 aggressively and accidentally freeze injected adapters too.
        # Re-assert trainability for the injected adapter modules here to avoid
        # "silent frozen adapters" in ablations.
        try:
            for p in self.srd_img_blocks.parameters():
                p.requires_grad_(True)
        except Exception:
            pass

        # --- memory SRD on mem_store_dim ---
        if self.mode == "ZERO_SHOT":
            self.srd_mem = None
        elif self.mode == "VANILLA":
            self.srd_mem = VanillaAdapter(d_model=self.mem_store_dim, m_rank=8)
        else:
            self.srd_mem = SRDAdapter(d_model=self.mem_store_dim, m_rank=8, use_fourier=False)

        # In SFA_ONLY baseline, temporal memory is disabled by design.
        # The memory-store sanitizer would be unused (no gradients / no effect),
        # so we freeze it to keep ablations clean and avoid "trainable-but-dead" params.
        if str(self.mode).upper() == "SFA_ONLY" and self.srd_mem is not None:
            try:
                for p in self.srd_mem.parameters():
                    p.requires_grad_(False)
            except Exception:
                pass

        # --- HR soil washers + BHFI ---
        if self.mode in ["FULL", "SFA_ONLY", "SFA_ASTA", "SRD_ONLY", "SRD_KMR"]:
            # alpha = sigmoid(raw) in (0,1); init raw=-6 => alpha≈0.0025 (near-identity)
            self.srd_washer_g1_raw = nn.Parameter(torch.full((1, 64, 1, 1), -6.0))
            self.srd_washer_g0_raw = nn.Parameter(torch.full((1, 32, 1, 1), -6.0))
            self.bhfi = BHFI(c0=32, c1=64, init_beta_raw=-6.0)
        else:
            self.srd_washer_g1_raw = None
            self.srd_washer_g0_raw = None
            self.bhfi = None

    # ---------------------------------------------------------------------
    # SRD injection helpers
    # ---------------------------------------------------------------------
    def _get_blocks(self, sam2):
        trunk = getattr(getattr(sam2, "image_encoder", None), "trunk", getattr(sam2, "image_encoder", None))
        if trunk is None:
            raise AttributeError("SAM2 has no image_encoder/trunk.")
        blocks = getattr(trunk, "blocks", None)
        if blocks is None:
            raise AttributeError("SAM2 image encoder has no blocks.")
        return blocks

    def _infer_block_dim(self, blk) -> int:
        for ln_name in ["norm2", "norm1", "norm"]:
            ln = getattr(blk, ln_name, None)
            if ln is not None and hasattr(ln, "normalized_shape"):
                ns = ln.normalized_shape
                if isinstance(ns, (tuple, list)) and len(ns) > 0:
                    return int(ns[-1])
                return int(ns)
        mlp = getattr(blk, "mlp", None)
        if mlp is not None:
            for attr in ["fc1", "w1", "lin1"]:
                layer = getattr(mlp, attr, None)
                if layer is not None and hasattr(layer, "in_features"):
                    return int(layer.in_features)
        return int(getattr(blk, "dim", 1024))

    def _inject_into_image_encoder(self, sam2):
        blocks = self._get_blocks(sam2)
        block_dims = [self._infer_block_dim(b) for b in blocks]

        # physical stage index by dim jumps
        stage_indices = []
        cur_stage, last_dim = -1, None
        for d in block_dims:
            if last_dim is None or d != last_dim:
                cur_stage += 1
                last_dim = d
            stage_indices.append(cur_stage)

        for i, blk in enumerate(blocks):
            dim = block_dims[i]
            if self.mode == "ZERO_SHOT":
                adapter = None
            elif self.mode == "VANILLA":
                adapter = VanillaAdapter(d_model=dim, m_rank=4)
            else:
                adapter = SRDAdapter(d_model=dim, m_rank=4, use_fourier=(stage_indices[i] <= 1))

            # Defensive: make sure newly created adapters are trainable.
            if adapter is not None:
                try:
                    for p in adapter.parameters():
                        p.requires_grad_(True)
                except Exception:
                    pass

            self.srd_img_blocks.append(adapter if adapter is not None else nn.Identity())

            if hasattr(blk, "mlp"):
                blk.mlp = SRDSafeWrapper(blk.mlp, adapter)

    # ---------------------------------------------------------------------
    # Interfaces used by RootSAMPro
    # ---------------------------------------------------------------------
    def decode_extras(self):
        """Extra kwargs for decode_masks_compat."""
        return dict(
            srd_washer_g0_raw=self.srd_washer_g0_raw,
            srd_washer_g1_raw=self.srd_washer_g1_raw,
            bhfi=self.bhfi,
        )

    def enhance_memory_store(self, F_mem_raw: torch.Tensor) -> torch.Tensor:
        """Apply SRD/Vanilla adapter on memory store maps (B, C=mem_store_dim, H, W)."""
        if self.srd_mem is None:
            return F_mem_raw
        if not (torch.is_tensor(F_mem_raw) and F_mem_raw.dim() == 4):
            return F_mem_raw

        Bm, Cm, Hm, Wm = F_mem_raw.shape
        x = F_mem_raw.reshape(Bm, Cm, -1).transpose(1, 2)  # (B,L,C)
        # adapter returns residual, we add it (safe)
        x = (x + self.srd_mem(x, hw_shape=(Hm, Wm))).transpose(1, 2).reshape(Bm, Cm, Hm, Wm)
        return x