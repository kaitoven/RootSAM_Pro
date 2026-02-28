import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
except Exception:
    build_sam2, build_sam2_video_predictor = None, None

from .adapters import SFAAdapter, ASTAAdapter, PRAAdapter, KMRAdapter
from .modules import DualMemoryBank, decode_masks_compat, detect_memory_dims


class RootSAMPro(nn.Module):
    """RootSAM-Pro (Tri-Adapter paradigm).

    - SFA : SRD(Image Encoder adapters) + BHFI + HR soil washers
    - ASTA: temporal read/valve-fuse/router/write (SFA_ASTA/FULL only)
    - PRA : reflex/presence (FULL only)
    """

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        m = str(mode).strip().upper()
        alias = {"SRD_ONLY": "SFA_ONLY", "SRD_KMR": "SFA_ASTA"}
        return alias.get(m, m)

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool):
        if module is None:
            return
        for p in module.parameters():
            try:
                p.requires_grad_(bool(trainable))
            except Exception:
                p.requires_grad = bool(trainable)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mode = self._normalize_mode(getattr(cfg, "ABLATION_MODE", "FULL"))

        if build_sam2 is None:
            raise RuntimeError("sam2 is not installed properly (cannot import build_sam2).")

        # ----------------------------
        # Build SAM2.1 and freeze base
        # ----------------------------
        try:
            self.sam2 = build_sam2(cfg.SAM2_MODEL_CFG, cfg.SAM2_CHECKPOINT)
        except Exception:
            self.sam2 = build_sam2_video_predictor(cfg.SAM2_MODEL_CFG, cfg.SAM2_CHECKPOINT)

        self.sam2.eval()
        for p in self.sam2.parameters():
            p.requires_grad = False

        # ----------------------------
        # Dimension radar
        # ----------------------------
        dims = detect_memory_dims(
            self.sam2,
            attn_d_model_default=256,
            mem_store_dim_default=64,
            ptr_dim_default=256,
            kv_in_dim_default=64,
        )
        self.attn_d_model = int(dims.attn_d_model)
        self.mem_store_dim = int(dims.mem_store_dim)
        self.ptr_dim = int(dims.ptr_dim)
        self.attn_kv_in_dim = int(dims.kv_in_dim)

        # ----------------------------
        # Adapters
        # ----------------------------
        self.sfa = SFAAdapter(cfg, self.mode, self.sam2, attn_d_model=self.attn_d_model, mem_store_dim=self.mem_store_dim)

        self.asta = ASTAAdapter(
            cfg, self.mode, self.sam2,
            attn_d_model=self.attn_d_model,
            mem_store_dim=self.mem_store_dim,
            ptr_dim=self.ptr_dim,
            kv_in_dim=self.attn_kv_in_dim,
        )

        self.pra = PRAAdapter(cfg, self.mode)
        # Optional KMR: kinematic refinement + probe head (improves clDice/Recall on gap/dynamic)
        kmr_enabled = bool(getattr(cfg, "KMR_ENABLED", True))
        if kmr_enabled and str(self.mode).strip().upper() in ("SFA_ASTA", "FULL"):
            kmr_m = int(getattr(cfg, "KMR_M", 32))
            kmr_n = int(getattr(cfg, "KMR_NUM_PROTOS", 64))
            self.kmr = KMRAdapter(d_model=self.attn_d_model, m_attn=kmr_m, num_prototypes=kmr_n)
        else:
            self.kmr = None

        # Apply strict ablation discipline (freeze modules)
        self._apply_ablation_trainability()

        # CRITICAL: SRD(Image) adapters live under sam2.*.mlp.adapter.*
        # We must re-enable them for SFA modes, otherwise SFA_ONLY becomes "BHFI-only".
        self._ensure_srd_image_trainable()

    def _apply_ablation_trainability(self):
        mode_u = str(self.mode).strip().upper()

        # PRA: FULL-only
        if mode_u != "FULL":
            self._set_module_trainable(self.pra, False)
            if hasattr(self.pra, "enabled"):
                self.pra.enabled = False

        # ASTA: only for SFA_ASTA / FULL
        if mode_u == "SFA_ONLY":
            self._set_module_trainable(self.asta, False)
            for k in ["enabled", "use_temporal_memory", "use_router"]:
                if hasattr(self.asta, k):
                    setattr(self.asta, k, False)

        # KMR: only meaningful when temporal engine is on
        if mode_u not in ("SFA_ASTA", "FULL"):
            self._set_module_trainable(self.kmr, False)

    def _ensure_srd_image_trainable(self):
        """Unfreeze injected SRD(Image) adapters inside SAM2.

        They appear as parameters named like:
          sam2.image_encoder.trunk.blocks.X.mlp.adapter.*
        """
        mode_u = str(self.mode).strip().upper()
        if mode_u not in ("SFA_ONLY", "SFA_ASTA", "FULL"):
            return

        # Unfreeze ONLY injected MLP adapters (not the whole SAM2).
        for name, p in self.sam2.named_parameters():
            if ".mlp.adapter." in name:
                try:
                    p.requires_grad_(True)
                except Exception:
                    p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # keep sam2 frozen + eval always
        if hasattr(self, "sam2") and self.sam2 is not None:
            self.sam2.eval()
        return self

    # ----------------------------
    # SAM2 module getters
    # ----------------------------
    def _get_prompt_encoder(self):
        pe = getattr(self.sam2, "prompt_encoder", getattr(self.sam2, "sam_prompt_encoder", None))
        if pe is None:
            raise AttributeError("SAM2 has no prompt encoder.")
        return pe

    # ----------------------------
    # Utilities
    # ----------------------------
    def _ensure_delta_t(self, delta_t, B: int, device, dtype):
        if torch.is_tensor(delta_t):
            dt = delta_t.to(device=device, dtype=dtype)
            if dt.dim() == 0:
                return dt.view(1).repeat(B)
            if dt.dim() == 1 and dt.shape[0] == 1 and B > 1:
                return dt.repeat(B)
            if dt.dim() == 1 and dt.shape[0] != B:
                return dt.view(-1)[:1].repeat(B)
            return dt
        if isinstance(delta_t, (float, int)):
            return torch.full((B,), float(delta_t), device=device, dtype=dtype)
        return torch.zeros((B,), device=device, dtype=dtype)

    def _collect_fpn_feats_bchw(self, high_res_features):
        feats = []
        if isinstance(high_res_features, dict):
            values = list(high_res_features.values())
        elif isinstance(high_res_features, (list, tuple)):
            values = list(high_res_features)
        else:
            values = []

        vals4 = [v for v in values if torch.is_tensor(v) and v.dim() == 4]
        if len(vals4) <= 1:
            layout = 'bchw'
            if len(vals4) == 1:
                v = vals4[0]
                cset = (8,16,24,32,48,64,80,96,112,128,160,192,224,256,320,384,512,640,768,1024)
                if int(v.shape[-1]) in cset and int(v.shape[1]) not in cset:
                    layout = 'bhwc'
        else:
            cset = (8,16,24,32,48,64,80,96,112,128,160,192,224,256,320,384,512,640,768,1024)

            def score(assume: str) -> float:
                items = []
                for v in vals4:
                    if assume == 'bchw':
                        c = int(v.shape[1]); area = int(v.shape[2]) * int(v.shape[3])
                    else:
                        c = int(v.shape[3]); area = int(v.shape[1]) * int(v.shape[2])
                    items.append((area, c))
                items.sort(key=lambda x: x[0], reverse=True)
                vio = 0
                invalid = 0
                prev = items[0][1]
                miss = sum(1 for _, c in items if c not in cset)
                if prev <= 0 or prev > 2048: invalid += 1
                for _, c in items[1:]:
                    if c < prev: vio += 1
                    if c <= 0 or c > 2048: invalid += 1
                    prev = c
                return float(vio) + 0.25*float(invalid) + 0.05*float(miss)

            layout = 'bhwc' if score('bhwc') < score('bchw') else 'bchw'

        for v in values:
            if not (torch.is_tensor(v) and v.dim() == 4):
                continue
            if layout == 'bhwc':
                v = v.permute(0, 3, 1, 2).contiguous()
            feats.append(v)
        return feats

    def _select_fpn_256_highres(self, high_res_features):
        feats = self._collect_fpn_feats_bchw(high_res_features)
        c256 = [f for f in feats if int(f.shape[1]) == int(self.attn_d_model)]
        if len(c256) == 0:
            return None
        return sorted(c256, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]), reverse=True)[0]

    # ----------------------------
    # Prompting & decoding
    # ----------------------------
    def _generate_dummy_prompts(self, B: int, H_feat: int, W_feat: int, device, dtype):
        prompt_encoder = self._get_prompt_encoder()
        empty_pts = torch.empty((B, 0, 2), device=device, dtype=dtype)
        empty_lbl = torch.empty((B, 0), device=device, dtype=dtype)
        try:
            sparse_embeddings, dense_embeddings = prompt_encoder(
                points=(empty_pts, empty_lbl),
                boxes=None,
                masks=None,
            )
        except Exception:
            sparse_embeddings = torch.zeros((B, 0, self.attn_d_model), device=device, dtype=dtype)
            dense_embeddings = torch.zeros((B, self.attn_d_model, H_feat, W_feat), device=device, dtype=dtype)

        if hasattr(prompt_encoder, "no_mask_embed") and hasattr(prompt_encoder.no_mask_embed, "weight"):
            try:
                w = prompt_encoder.no_mask_embed.weight
                dense = w.view(1, -1, 1, 1).to(device=device, dtype=dtype).repeat(B, 1, H_feat, W_feat)
                dense_embeddings = dense
            except Exception:
                pass

        return sparse_embeddings, dense_embeddings

    def _extract_obj_pointer(self, dec_out, feat_bchw: torch.Tensor, mask_logits_b1hw: torch.Tensor) -> torch.Tensor | None:
        if isinstance(dec_out, (tuple, list)):
            for x in dec_out:
                if torch.is_tensor(x) and x.dim() == 3 and x.shape[-1] == self.ptr_dim and x.shape[1] >= 1:
                    return x[:, 0:1, :].contiguous()
        if isinstance(dec_out, dict):
            for k in ["obj_ptr", "obj_ptrs", "object_ptr", "object_ptrs", "obj_ptr_tks", "obj_ptr_tokens"]:
                if k in dec_out and torch.is_tensor(dec_out[k]) and dec_out[k].dim() == 3 and dec_out[k].shape[-1] == self.ptr_dim:
                    x = dec_out[k]
                    if x.shape[1] >= 1:
                        return x[:, 0:1, :].contiguous()

        with torch.no_grad():
            m = torch.sigmoid(mask_logits_b1hw)
            m = F.interpolate(m, size=feat_bchw.shape[-2:], mode="bilinear", align_corners=False)
            w = m / (m.sum(dim=(2, 3), keepdim=True) + 1e-6)
            ptr = (feat_bchw * w).sum(dim=(2, 3), keepdim=False)
            return ptr.view(ptr.shape[0], 1, ptr.shape[1]).contiguous()

    # ----------------------------
    # Forward: one TBPTT step
    # ----------------------------
    def forward(self, image_t, delta_t, inference_state_in, frame_idx):
        B = int(image_t.shape[0])
        device, target_dtype = image_t.device, image_t.dtype
        dt = self._ensure_delta_t(delta_t, B, device, dtype=torch.float32)

        bank_old = DualMemoryBank.read(inference_state_in, B=B, device=device, frame_idx=int(frame_idx))
        old_time_days = bank_old.time_days
        old_is_mem_empty = bank_old.is_mem_empty

        dt_eff = torch.where(dt < 900.0, dt, torch.zeros_like(dt)).float()
        curr_time_days = old_time_days + dt_eff

        backbone_out = self.sam2.forward_image(image_t)
        high_res_features = backbone_out.get("backbone_fpn", None)
        F_base = self._select_fpn_256_highres(high_res_features)
        if F_base is None:
            raise RuntimeError("backbone_fpn is missing valid C=256 4D features.")
        Hf, Wf = int(F_base.shape[-2]), int(F_base.shape[-1])

        mode_u = str(self.mode).strip().upper()
        use_asta = (mode_u in ("SFA_ASTA", "FULL")) and bool(getattr(self.asta, "enabled", False))

        # 1) ASTA read (SAM2 memory attention -> context)
        if use_asta:
            F_attn, sim, _ = self.asta.temporal_read(
                F_base=F_base,
                bank_old=bank_old,
                curr_time_days=curr_time_days,
                target_hw=(Hf, Wf),
                debug_events=None,
            )
        else:
            F_attn = F_base
            sim = torch.ones(B, device=device, dtype=torch.float32)

        # 1.5) ASTA-V1 ReZero injection (feature-level; starts as identity)
        if use_asta and bool(getattr(self.asta, "asta_v1_enabled", False)) and (getattr(self.asta, "asta_gamma", None) is not None):
            gamma = self.asta.asta_gamma.to(device=device, dtype=target_dtype)
            F_track = F_base + gamma * (F_attn.to(dtype=target_dtype) - F_base)
        else:
            F_track = F_attn.to(dtype=target_dtype) if use_asta else F_base

        # 1.6) KMR (optional kinematic refinement + probe head)
        use_kmr = (self.kmr is not None) and bool(getattr(self.cfg, "KMR_ENABLED", True)) and (mode_u in ("SFA_ASTA", "FULL"))
        if use_kmr:
            F_track, probes, F_kin = self.kmr(F_track, dt_eff.to(device=device, dtype=target_dtype))
        else:
            probes = torch.zeros((B, 3, Hf, Wf), device=device, dtype=target_dtype)
            F_kin = torch.ones((B, 1, 1, 1), device=device, dtype=target_dtype)

        # 2) decode (prompt-free) on tracked features
        F_route = F_track

        sparse_pe, dense_pe = self._generate_dummy_prompts(B, int(F_route.shape[2]), int(F_route.shape[3]), device, target_dtype)

        prompt_encoder = self._get_prompt_encoder()
        if hasattr(prompt_encoder, "get_dense_pe"):
            curr_pe = prompt_encoder.get_dense_pe().to(device=device, dtype=target_dtype)
            if curr_pe.size(0) != 1:
                curr_pe = curr_pe[:1]
            if curr_pe.shape[-2:] != F_route.shape[-2:]:
                curr_pe = F.interpolate(curr_pe, size=F_route.shape[-2:], mode="bilinear", align_corners=False)
        else:
            curr_pe = torch.zeros_like(dense_pe[:1])

        dec_out = decode_masks_compat(
            sam2=self.sam2,
            image_embeddings=F_route,
            image_pe=curr_pe,
            sparse_prompt_embeddings=sparse_pe,
            dense_prompt_embeddings=dense_pe,
            multimask_output=False,
            high_res_features=high_res_features,
            **self.sfa.decode_extras(),
        )
        M_logits = (dec_out[0] if isinstance(dec_out, (tuple, list)) else dec_out)[:, 0:1, :, :]
        M_logits_base = M_logits

        # 3) ASTA fuse
        if use_asta:
            M_logits_asta, valve_gate_loss = self.asta.fuse_logits_oneway(
                F_base=F_base,
                F_attn=F_attn,
                logits_base_b1hw=M_logits_base,
                dt_eff=dt_eff,
                sim=sim,
            )
        else:
            M_logits_asta, valve_gate_loss = M_logits_base, None

        obj_ptr_decoder = self._extract_obj_pointer(dec_out, F_route, M_logits_base)

        # 4) PRA FULL-only
        Pc_prob, p_present = self.pra.compute_presence_proxy(self.mode, M_logits_asta, probes)

        pra_enabled = bool(getattr(self.pra, "enabled", False)) and (mode_u == "FULL")
        if pra_enabled:
            pra_out = self.pra(M_logits_asta, Pc_prob, p_present)
        else:
            pra_out = (M_logits_asta, torch.zeros(B, dtype=torch.bool, device=device))

        if isinstance(pra_out, (tuple, list)):
            M_fused_logits = pra_out[0]
            reset_flags = pra_out[1] if len(pra_out) > 1 else torch.zeros(B, dtype=torch.bool, device=device)
        else:
            M_fused_logits = pra_out
            reset_flags = torch.zeros(B, dtype=torch.bool, device=device)

        # 5) ASTA router/write only when enabled
        if use_asta:
            # Important: router/gate loss should NOT backprop into segmentation logits (prevents weird coupling)
            logits_for_router = M_logits_asta.detach() if torch.is_tensor(M_logits_asta) else M_logits_asta
            obj_ptr_for_router = obj_ptr_decoder.detach() if torch.is_tensor(obj_ptr_decoder) else obj_ptr_decoder
            write_gate, key_weight, val_prob2, gate_loss = self.asta.route(
                dt_eff=dt_eff,
                sim=sim,
                p_present=p_present,
                bank_old=bank_old,
                logits_b1hw=logits_for_router,
                obj_ptr_decoder=obj_ptr_for_router,
                reset_flags=reset_flags,
            )
        else:
            write_gate = torch.zeros((B,), device=device, dtype=torch.float32)
            key_weight = torch.zeros((B,), device=device, dtype=torch.float32)
            val_prob2 = torch.full((B, 2), 0.5, device=device, dtype=torch.float32)
            gate_loss = torch.zeros((), device=device, dtype=torch.float32)

        gate_loss_total = gate_loss
        if valve_gate_loss is not None and torch.is_tensor(valve_gate_loss):
            gate_loss_total = valve_gate_loss if gate_loss_total is None else (gate_loss_total + valve_gate_loss)

        min_w = float(getattr(self.cfg, "MEMENC_MIN_WRITE_GATE", 1e-6))
        did_write = ((write_gate.detach().float() * p_present.detach().float()) > min_w)
        new_is_mem_empty = torch.where(
            reset_flags.to(dtype=torch.bool),
            torch.ones_like(did_write, dtype=torch.bool, device=device),
            torch.where(did_write.to(dtype=torch.bool), torch.zeros_like(did_write, dtype=torch.bool, device=device), old_is_mem_empty.to(dtype=torch.bool)),
        )

        if use_asta:
            new_state = self.asta.write_state(
                bank_old=bank_old,
                frame_idx=int(frame_idx),
                curr_time_days=curr_time_days,
                F_base=F_base,
                M_fused_logits=M_fused_logits,
                obj_ptr_decoder=obj_ptr_for_router,
                write_gate=write_gate,
                key_weight=key_weight,
                val_prob2=val_prob2,
                p_present=p_present,
                reset_flags=reset_flags,
                new_is_mem_empty=new_is_mem_empty,
                target_dtype=target_dtype,
                debug_events=None,
                mem_sanitizer=self.sfa.enhance_memory_store,
            )
        else:
            new_state = {}

        is_training = bool(self.training and torch.is_grad_enabled())
        if is_training:
            return M_fused_logits, probes, F_kin, F_track, new_state, gate_loss_total
        return M_fused_logits, probes, F_kin, F_track, new_state