import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def autocast_cuda(enabled: bool = True):
    """
    Compatibility wrapper:
    - New API: torch.amp.autocast("cuda", ...)
    - Old API: torch.cuda.amp.autocast(...)
    """
    try:
        # PyTorch >= 2.0
        return torch.amp.autocast("cuda", enabled=enabled)
    except Exception:
        # PyTorch < 2.0
        return torch.cuda.amp.autocast(enabled=enabled)

from ..modules import (
    DualMemoryBank,
    MemoryRouter,
    memory_attention_compat,
    has_official_obj_ptr_proj,
    get_official_obj_ptr_proj,
)


class ASTAAdapter(nn.Module):
    """ASTA: Adaptive Spatio-Temporal Adapter (Memory Attention + Router + DualBank + Write-back).

    This adapter encapsulates:
      - strict ablation discipline for temporal engine
      - optional ASTA-V1 (Vacuum Bypass + ReZero residual injection)
      - keyframe router + write-back scheduling
      - memory encoder write-back with P2 mask-grad option (weights frozen)

    It is designed to be plug-and-play: RootSAMPro can enable/disable ASTA
    based on `ABLATION_MODE` and strict flags in cfg.
    """

    def __init__(self, cfg, mode: str, sam2, attn_d_model: int, mem_store_dim: int, ptr_dim: int, kv_in_dim: int):
        super().__init__()
        self.cfg = cfg
        self.mode = str(mode)
        self.sam2 = sam2

        self.attn_d_model = int(attn_d_model)
        self.mem_store_dim = int(mem_store_dim)
        self.ptr_dim = int(ptr_dim)
        self.attn_kv_in_dim = int(kv_in_dim)

        # ------------------------------------------------------------
        # P0: Strict ablation discipline switches (orthogonal ablations)
        # ------------------------------------------------------------
        self._strict_no_memory = bool(getattr(cfg, "ABLATION_STRICT_NO_MEMORY", True))
        self._strict_no_router = bool(getattr(cfg, "ABLATION_STRICT_NO_ROUTER", True))
        temporal_modes = getattr(cfg, "TEMPORAL_ENGINE_MODES", ["SFA_ASTA", "FULL"])
        if isinstance(temporal_modes, (tuple, list, set)):
            temporal_modes = set(list(temporal_modes))
        else:
            temporal_modes = set([str(temporal_modes)])

        self.use_temporal_memory = (self.mode in temporal_modes) if self._strict_no_memory else True
        self.use_router = (self.mode in temporal_modes) if self._strict_no_router else True

        # For paper-level notion: ASTA is considered "enabled" when temporal memory is allowed.
        self.enabled = bool(self.use_temporal_memory)

        # Dual FIFO + TBPTT configs (owned by ASTA)
        self.tbptt_keep_last = int(getattr(cfg, "TBPTT_KEEP_LAST", 2))
        self.max_recent_frames = int(getattr(cfg, "MAX_RECENT_FRAMES", int(getattr(cfg, "MAX_MEM_FRAMES", 5))))
        self.max_prompted_frames = int(getattr(cfg, "MAX_PROMPTED_FRAMES", 2))

        self.mem_pool_stride = int(getattr(cfg, "MEM_POOL_STRIDE", 1))
        self.use_temporal_pos = bool(getattr(cfg, "USE_TEMPORAL_POS", True))
        self.use_bio_kes = bool(getattr(cfg, "USE_BIO_KES", True))
        self.add_tpos_to_ptrs = bool(getattr(cfg, "ADD_TPOS_TO_PTRS", False))

        # Bio-KES half-lives (days)
        init_half_life_days = 180.0
        hl_raw0 = float(torch.log(torch.expm1(torch.tensor(init_half_life_days))).item())  # inv-softplus
        self.bio_half_life_raw = nn.Parameter(torch.full((1, 1, self.attn_kv_in_dim), hl_raw0))
        self.ptr_half_life_raw = nn.Parameter(torch.full((1, 1, self.attn_kv_in_dim), hl_raw0))

        # value half-life (days)
        init_val_hl_days = 180.0
        val_hl_raw0 = float(torch.log(torch.expm1(torch.tensor(init_val_hl_days))).item())  # inv-softplus
        self.value_half_life_raw = nn.Parameter(torch.tensor(val_hl_raw0))

        # Projections for memory packing (trainable, small)
        self.srd_mem_proj = nn.Identity()
        if self.mem_store_dim != self.attn_kv_in_dim:
            self.srd_mem_proj = nn.Linear(self.mem_store_dim, self.attn_kv_in_dim, bias=False)
            nn.init.normal_(self.srd_mem_proj.weight, std=0.02)

        self.srd_ptr_proj = nn.Identity()
        if self.ptr_dim != self.attn_kv_in_dim:
            self.srd_ptr_proj = nn.Linear(self.ptr_dim, self.attn_kv_in_dim, bias=False)
            nn.init.normal_(self.srd_ptr_proj.weight, std=0.02)

        self.has_official_ptr_proj = bool(has_official_obj_ptr_proj(self.sam2))

        # Router module
        self.router = MemoryRouter() if self.use_router else None

        # ------------------------------------------------------------
        # ASTA-V1: Vacuum bypass + ReZero residual injection
        # ------------------------------------------------------------
        self.asta_v1_enabled = bool(getattr(cfg, "ASTA_V1_ENABLED", True))
        self.asta_v1_vacuum_bypass = bool(getattr(cfg, "ASTA_V1_VACUUM_BYPASS", True))
        if self.asta_v1_enabled:
            g0 = float(getattr(cfg, "ASTA_V1_GAMMA_INIT", 0.0))
            self.asta_gamma = nn.Parameter(torch.full((1, self.attn_d_model, 1, 1), g0, dtype=torch.float32))
            if not bool(self.use_temporal_memory):
                self.asta_gamma.requires_grad_(False)
        else:
            self.asta_gamma = None

        # ------------------------------------------------------------
        # OTACF-style logit one-way valve (B-route: memory_attention -> Δ_logits)
        #   M_fused = M_base + alpha * ReLU(delta_logits)
        # ------------------------------------------------------------
        # Pixel-wise entropy gate alpha
        self.alpha_tau_raw = nn.Parameter(torch.tensor(0.35))      # passed through sigmoid -> (0,1)
        self.alpha_temp_raw = nn.Parameter(torch.tensor(-1.0))     # softplus -> temperature
        self.alpha_l1_raw = nn.Parameter(torch.tensor(-6.9))       # softplus -> sparsity weight (~0.001)

        self.valve_l1_weight = float(getattr(cfg, "VALVE_L1_WEIGHT", 1e-3))
        # optional presence-gate for alpha (reduces soil FP under one-way valve)
        self.alpha_pres_tau_raw = nn.Parameter(torch.tensor(-1.4, dtype=torch.float32))  # sigmoid->~0.20
        self.alpha_pres_temp_raw = nn.Parameter(torch.tensor(-2.3, dtype=torch.float32)) # softplus->~0.10

        # Δ logits head: use (F_attn - F_base) reduced + [entropy, dt, sim]
        self.delta_reduce = nn.Conv2d(self.attn_d_model, 32, kernel_size=1)
        self.delta_head = nn.Sequential(
            nn.Conv2d(32 + 1 + 1 + 1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

        # Freeze temporal knobs when temporal memory is disabled (strict ablation discipline)
        if not self.use_temporal_memory:
            for p in [self.bio_half_life_raw, self.ptr_half_life_raw, self.value_half_life_raw]:
                try:
                    p.requires_grad_(False)
                except Exception:
                    pass
            # Hard-freeze ALL ASTA parameters when temporal engine is disabled
            for p in self.parameters():
                try:
                    p.requires_grad_(False)
                except Exception:
                    pass

    # ---------------------------------------------------------------------
    # Utilities expected by pack_memory_bank / memory_attention_compat
    # ---------------------------------------------------------------------
    def _pool_mem_map(self, v_bchw: torch.Tensor) -> torch.Tensor:
        s = int(self.mem_pool_stride)
        if s <= 1:
            return v_bchw
        return F.avg_pool2d(v_bchw, kernel_size=s, stride=s, padding=0)

    def _to_tokens_bLc(self, x_bchw: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_bchw.shape
        return x_bchw.flatten(2).transpose(1, 2).contiguous()

    def _project_ptr_to_kv(self, obj_ptr: torch.Tensor) -> torch.Tensor:
        """Project raw obj pointer (B,1,ptr_dim) -> (B,1,kv_in_dim)."""
        if obj_ptr is None:
            return None
        if self.has_official_ptr_proj:
            proj = get_official_obj_ptr_proj(self.sam2)
            if proj is not None:
                try:
                    return proj(obj_ptr)
                except Exception:
                    pass
        # fallback: trainable linear projection (often identity if dims match)
        if isinstance(self.srd_ptr_proj, nn.Identity):
            return obj_ptr
        # nn.Linear expects (..., in_features)
        return self.srd_ptr_proj(obj_ptr)

    def _get_memory_encoder(self):
        me = getattr(self.sam2, "memory_encoder", getattr(self.sam2, "sam_memory_encoder", None))
        if me is None:
            raise AttributeError("SAM2 has no memory encoder.")
        return me

    # ---------------------------------------------------------------------
    # Public API used by RootSAMPro
    # ---------------------------------------------------------------------
    def temporal_read(self, F_base: torch.Tensor, bank_old, curr_time_days, target_hw, debug_events=None):
        """Read temporal context via SAM2 memory attention (B-route).

        Returns:
          F_attn: (B, C, H, W) context feature map (falls back to F_base when memory empty)
          sim: (B,) similarity scalar in [0,1]
          debug_events: passthrough
        """
        B = int(F_base.shape[0])
        device = F_base.device

        if not self.use_temporal_memory:
            return F_base, torch.ones(B, device=device, dtype=torch.float32), debug_events

        old_is_mem_empty = bank_old.is_mem_empty

        # Vacuum bypass if all samples are empty (cheap fast-path)
        if self.asta_v1_enabled and self.asta_v1_vacuum_bypass and bool(old_is_mem_empty.all().item()):
            return F_base, torch.ones(B, device=device, dtype=torch.float32), debug_events

        # with torch.cuda.amp.autocast(enabled=False):
        with autocast_cuda(enabled=False):
            F_attn, sim, debug_events = memory_attention_compat(
                host=self,
                F_base=F_base.float(),
                bank_old=bank_old,
                curr_time_days=curr_time_days,
                target_hw=target_hw,
                debug_events=debug_events,
            )
        # sanitize numerics (AMP / attention can occasionally produce NaN/Inf)
        F_attn = torch.nan_to_num(F_attn, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=F_base.dtype)
        sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0).to(device=device, dtype=torch.float32)
        return F_attn, sim, debug_events

    @staticmethod
    def _entropy_from_logits(logits_b1hw: torch.Tensor) -> torch.Tensor:
        """Pixel-wise Shannon entropy of sigmoid(logits), normalized to [0,1]."""
        p = torch.sigmoid(logits_b1hw).clamp(1e-6, 1.0 - 1e-6)
        H = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        H = H / math.log(2.0)
        return H.clamp(0.0, 1.0)

    def fuse_logits_oneway(
        self,
        *,
        F_base: torch.Tensor,
        F_attn: torch.Tensor,
        logits_base_b1hw: torch.Tensor,
        dt_eff: torch.Tensor,
        sim: torch.Tensor,
    ):
        """OTACF one-way valve fusion (logit space).

        Δ_logits is predicted from memory_attention context (F_attn) and fused as:
            logits_fused = logits_base + alpha * ReLU(Δ_logits)

        Notes:
          - All critical math runs in fp32 to avoid AMP NaN/Inf.
          - Optional presence-gate further suppresses soil FP under one-way valve.
        """
        if not self.use_temporal_memory:
            return logits_base_b1hw, None

        B = int(logits_base_b1hw.shape[0])
        device = logits_base_b1hw.device
        Hm, Wm = int(logits_base_b1hw.shape[-2]), int(logits_base_b1hw.shape[-1])

        # normalize dt (treat NA sentinel like 0.0)
        if not torch.is_tensor(dt_eff):
            dt = torch.full((B,), float(dt_eff), device=device, dtype=torch.float32)
        else:
            dt = dt_eff.to(device=device, dtype=torch.float32).view(-1)
            if dt.numel() == 1:
                dt = dt.repeat(B)
        # PRMI: first frame often uses 999 as NA; do not let it dominate
        dt = torch.where(dt > 900.0, torch.zeros_like(dt), dt)

        # with torch.cuda.amp.autocast(enabled=False):
        with autocast_cuda(enabled=False):
            logits32 = logits_base_b1hw.float()
            F_base32 = F_base.float()
            F_attn32 = F_attn.float()
            sim32 = sim.to(device=device, dtype=torch.float32).view(B, 1).clamp(0.0, 1.0)

            # entropy -> alpha (pixel-wise)
            entropy = self._entropy_from_logits(logits32)  # (B,1,Hm,Wm) float32
            tau = torch.sigmoid(self.alpha_tau_raw).to(device=device, dtype=torch.float32)
            temp = F.softplus(self.alpha_temp_raw).to(device=device, dtype=torch.float32) + 1e-6
            alpha = torch.sigmoid((entropy - tau) / temp)

            # optional presence gate (suppresses pure-soil FP)
            pres_tau = torch.sigmoid(self.alpha_pres_tau_raw).to(device=device, dtype=torch.float32)  # ~0.2 init
            pres_temp = F.softplus(self.alpha_pres_temp_raw).to(device=device, dtype=torch.float32) + 1e-6
            pres_map = torch.sigmoid(logits32)  # base prob
            pres_gate = torch.sigmoid((pres_map - pres_tau) / pres_temp)
            alpha = alpha * pres_gate

            # global presence gate (sequence-level; suppresses soil FP under one-way valve)
            g_tau = float(getattr(self.cfg, "VALVE_GLOBAL_PRES_TAU", 0.03))
            g_temp = float(getattr(self.cfg, "VALVE_GLOBAL_PRES_TEMP", 0.05))
            pres_scalar = pres_map.mean(dim=(2, 3), keepdim=True)  # (B,1,1,1)
            g_global = torch.sigmoid((pres_scalar - g_tau) / max(g_temp, 1e-6))
            alpha = alpha * g_global

            # Δ logits head input
            diff = (F_attn32 - F_base32)
            diff_r = self.delta_reduce(diff)  # (B,32,Hf,Wf)
            diff_up = F.interpolate(diff_r, size=(Hm, Wm), mode="bilinear", align_corners=False)

            # scalar maps
            dt_mean = dt.mean().clamp(min=1e-6)
            dt_map = (dt.view(B, 1, 1, 1) / dt_mean).clamp(0.0, 10.0).expand(B, 1, Hm, Wm)
            sim_map = sim32.view(B, 1, 1, 1).expand(B, 1, Hm, Wm)

            delta_in = torch.cat([diff_up, entropy, dt_map, sim_map], dim=1)
            delta_logits = self.delta_head(delta_in)  # (B,1,Hm,Wm)
            # cap delta magnitude (stabilize & reduce FP bursts)
            dcap = float(getattr(self.cfg, "DELTA_LOGITS_CAP", 5.0))
            if dcap > 0:
                delta_logits = torch.clamp(delta_logits, min=-dcap, max=dcap)

            # sanitize
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
            delta_logits = torch.nan_to_num(delta_logits, nan=0.0, posinf=0.0, neginf=0.0)

            logits_fused32 = logits32 + alpha * F.relu(delta_logits)

            # control sparsity loss on alpha (small weight)
            alpha_l1 = F.softplus(self.alpha_l1_raw).to(device=device, dtype=torch.float32)
            gate_loss = (self.valve_l1_weight * alpha_l1) * alpha.mean()

        return logits_fused32.to(dtype=logits_base_b1hw.dtype), gate_loss.to(dtype=torch.float32)

    def route(self, *, dt_eff, sim, p_present, bank_old, logits_b1hw, obj_ptr_decoder, reset_flags):
        """Run learned router (or safe deterministic fallback) to obtain write_gate and split."""
        B = int(logits_b1hw.shape[0])
        device = logits_b1hw.device
        gate_loss = torch.zeros((), device=device, dtype=torch.float32)

        if not self.use_temporal_memory:
            write_gate = torch.zeros(B, device=device, dtype=torch.float32)
            key_weight = torch.zeros(B, device=device, dtype=torch.float32)
            val_prob2 = torch.full((B, 2), 0.5, device=device, dtype=torch.float32)
            return write_gate, key_weight, val_prob2, gate_loss

        old_out_dict, old_p_out_dict = bank_old.out_dict, bank_old.p_out_dict

        if self.use_router and (self.router is not None):
            full_recent = bool(len(old_out_dict) >= int(self.max_recent_frames))
            full_prompted = bool(len(old_p_out_dict) >= int(self.max_prompted_frames))

            router_out = self.router(
                dt_days=dt_eff.to(device=device, dtype=torch.float32),
                sim=sim.to(device=device, dtype=torch.float32),
                p_present=p_present.to(device=device, dtype=torch.float32),
                old_is_mem_empty=bank_old.is_mem_empty,
                logits_b1hw=logits_b1hw,
                obj_ptr_decoder=obj_ptr_decoder,
                recent_ptrs=list(bank_old.obj_ptrs.values()),
                prompt_ptrs=list(bank_old.p_obj_ptrs.values()),
                full_recent=full_recent,
                full_prompted=full_prompted,
                reset_flags=reset_flags,
                mode=self.mode,
                ptr_half_life_raw=self.ptr_half_life_raw,
            )
            return router_out.write_gate, router_out.split_weight, router_out.val_prob2, router_out.gate_loss

        if self.use_router and (self.router is None):
            # Deterministic fallback (threshold-free)
            write_gate = (p_present * (~reset_flags).to(torch.float32)).clamp(0.0, 1.0)
            key_weight = torch.zeros_like(write_gate)
            val_prob2 = torch.stack([torch.ones_like(write_gate), torch.zeros_like(write_gate)], dim=1)
            return write_gate, key_weight, val_prob2, gate_loss

        # Temporal read-only: allow memory attention, but no write-back / no gate loss.
        write_gate = torch.zeros(B, device=device, dtype=torch.float32)
        key_weight = torch.zeros(B, device=device, dtype=torch.float32)
        val_prob2 = torch.full((B, 2), 0.5, device=device, dtype=torch.float32)
        return write_gate, key_weight, val_prob2, gate_loss

    def _extract_mem_map(self, mem_out, B, Hf, Wf, target_dtype):
        F_mem_raw = None
        if isinstance(mem_out, dict):
            for k in ["mem_features", "memory_features", "vision_features", "features", "mem", "memory"]:
                if k in mem_out and torch.is_tensor(mem_out[k]):
                    v = mem_out[k]
                    if v.dim() == 4 and int(v.shape[1]) == int(self.mem_store_dim):
                        F_mem_raw = v
                        break
                    if v.dim() == 4 and int(v.shape[-1]) == int(self.mem_store_dim):
                        F_mem_raw = v.permute(0, 3, 1, 2).contiguous()
                        break
        elif isinstance(mem_out, (tuple, list)) and len(mem_out) >= 1 and torch.is_tensor(mem_out[0]):
            v = mem_out[0]
            if v.dim() == 4 and int(v.shape[1]) == int(self.mem_store_dim):
                F_mem_raw = v
            elif v.dim() == 4 and int(v.shape[-1]) == int(self.mem_store_dim):
                F_mem_raw = v.permute(0, 3, 1, 2).contiguous()
        elif torch.is_tensor(mem_out) and mem_out.dim() == 4:
            if int(mem_out.shape[1]) == int(self.mem_store_dim):
                F_mem_raw = mem_out
            elif int(mem_out.shape[-1]) == int(self.mem_store_dim):
                F_mem_raw = mem_out.permute(0, 3, 1, 2).contiguous()

        if F_mem_raw is None:
            F_mem_raw = torch.zeros((B, self.mem_store_dim, Hf, Wf), device=self.sam2.device if hasattr(self.sam2,'device') else None, dtype=target_dtype)
        return F_mem_raw

    def write_state(
        self,
        *,
        bank_old,
        frame_idx: int,
        curr_time_days: torch.Tensor,
        F_base: torch.Tensor,
        M_fused_logits: torch.Tensor,
        obj_ptr_decoder: torch.Tensor,
        write_gate: torch.Tensor,
        key_weight: torch.Tensor,
        val_prob2: torch.Tensor,
        p_present: torch.Tensor,
        reset_flags: torch.Tensor,
        new_is_mem_empty: torch.Tensor,
        target_dtype,
        debug_events=None,
        mem_sanitizer=None,
    ):
        """Encode memory (optional) and write functional state via DualMemoryBank."""
        B = int(F_base.shape[0])
        Hf, Wf = int(F_base.shape[-2]), int(F_base.shape[-1])
        device = F_base.device

        mem_out = None
        if self.use_temporal_memory:
            min_w = float(getattr(self.cfg, "MEMENC_MIN_WRITE_GATE", 1e-6))
            # Avoid encoding/writing memory for absent-root frames by coupling with presence
            need_write = bool(((write_gate.detach() * p_present.detach()) > min_w).any().item())

            if need_write:
                memory_encoder = self._get_memory_encoder()
                allow_mask_grad = bool(
                    self.training and torch.is_grad_enabled() and bool(getattr(self.cfg, "ALLOW_MEMORY_MASK_GRAD", True))
                )

                pix_feat = F_base
                if bool(getattr(self.cfg, "DETACH_PIX_FEAT_IN_MEMENC", True)):
                    pix_feat = pix_feat.detach()

                for scale in [1, 4, 8, 16]:
                    m_logits = F.interpolate(
                        M_fused_logits,
                        size=(int(Hf * scale), int(Wf * scale)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    m = torch.sigmoid(m_logits).to(dtype=target_dtype)
                    if not allow_mask_grad:
                        m = m.detach()

                    try:
                        # with torch.cuda.amp.autocast(enabled=False):
                        with autocast_cuda(enabled=False):
                            mem_out = memory_encoder(pix_feat=pix_feat.float(), masks=m.float())
                        break
                    except TypeError:
                        try:
                            with torch.cuda.amp.autocast(enabled=False):
                                mem_out = memory_encoder(pix_feat.float(), m.float())
                            break
                        except RuntimeError as e:
                            if any(s in str(e).lower() for s in ["match", "size", "shape"]):
                                continue
                            raise e
                    except RuntimeError as e:
                        if any(s in str(e).lower() for s in ["match", "size", "shape"]):
                            continue
                        raise e

        # Extract mem map
        F_mem_raw = None
        if isinstance(mem_out, dict):
            for k in ["mem_features", "memory_features", "vision_features", "features", "mem", "memory"]:
                if k in mem_out and torch.is_tensor(mem_out[k]):
                    v = mem_out[k]
                    if v.dim() == 4 and int(v.shape[1]) == int(self.mem_store_dim):
                        F_mem_raw = v
                        break
                    if v.dim() == 4 and int(v.shape[-1]) == int(self.mem_store_dim):
                        F_mem_raw = v.permute(0, 3, 1, 2).contiguous()
                        break
        elif isinstance(mem_out, (tuple, list)) and len(mem_out) >= 1 and torch.is_tensor(mem_out[0]):
            v = mem_out[0]
            if v.dim() == 4 and int(v.shape[1]) == int(self.mem_store_dim):
                F_mem_raw = v
            elif v.dim() == 4 and int(v.shape[-1]) == int(self.mem_store_dim):
                F_mem_raw = v.permute(0, 3, 1, 2).contiguous()
        elif torch.is_tensor(mem_out) and mem_out.dim() == 4:
            if int(mem_out.shape[1]) == int(self.mem_store_dim):
                F_mem_raw = mem_out
            elif int(mem_out.shape[-1]) == int(self.mem_store_dim):
                F_mem_raw = mem_out.permute(0, 3, 1, 2).contiguous()

        if F_mem_raw is None:
            F_mem_raw = torch.zeros((B, self.mem_store_dim, Hf, Wf), device=device, dtype=target_dtype)
        # sanitize memory features
        F_mem_raw = torch.nan_to_num(F_mem_raw, nan=0.0, posinf=0.0, neginf=0.0)

        if F_mem_raw.shape[-2:] != (Hf, Wf):
            F_mem_raw = F.interpolate(F_mem_raw, size=(Hf, Wf), mode="bilinear", align_corners=False)

        if callable(mem_sanitizer):
            try:
                F_mem_raw = mem_sanitizer(F_mem_raw)
            except Exception:
                pass

        new_state = DualMemoryBank.write_new_state(
            old=bank_old,
            frame_idx=int(frame_idx),
            mem_feat_bchw=F_mem_raw,
            obj_ptr_decoder=obj_ptr_decoder,
            curr_time_days=curr_time_days,
            write_gate=write_gate,
            split_weight=key_weight,
            val_prob2=val_prob2,
            p_present=p_present,
            reset_flags=reset_flags,
            new_is_mem_empty=new_is_mem_empty,
            max_recent_frames=int(self.max_recent_frames),
            max_prompted_frames=int(self.max_prompted_frames),
            target_dtype=target_dtype,
            debug_events=debug_events,
        )
        return new_state
