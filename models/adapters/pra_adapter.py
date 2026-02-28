import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.gfu_firewall import ReflexMechanisms


class PRAAdapter(nn.Module):
    """PRA: Physical-Reflex Adapter.

    Encapsulates:
      - GFU refinement (guided fusion)
      - learned causal firewall
      - learned presence suppression (continuous, no heuristic thresholds)

    Losses like soft-clDice / DPSL++ are handled in criterion (engine/losses.py).
    This module provides the *reflex refinement* and the reset/flush signals.
    """

    def __init__(self, cfg, mode: str):
        super().__init__()
        self.cfg = cfg
        self.mode = str(mode)

        self.enabled = bool(self.mode == "FULL") and bool(getattr(cfg, "PRA_ENABLED", True))

        # Parameters (learned gates, no hand thresholds)
        self.kappa_raw = nn.Parameter(torch.tensor(0.0))

        # presence suppressor
        init_tau = 0.05
        init_temp = 0.02
        init_bias_mag = 4.0
        self.pres_tau_logit = nn.Parameter(torch.tensor(math.log(init_tau / max(1e-6, (1.0 - init_tau)))))
        temp_raw0 = float(torch.log(torch.expm1(torch.tensor(init_temp))).item())  # inv-softplus
        bias_raw0 = float(torch.log(torch.expm1(torch.tensor(init_bias_mag))).item())  # inv-softplus
        self.pres_temp_raw = nn.Parameter(torch.tensor(temp_raw0))
        self.abs_bias_mag_raw = nn.Parameter(torch.tensor(bias_raw0))

        # firewall thresholds (learned)
        init_eta = 50.0
        init_theta = 0.8
        eta_raw0 = float(torch.log(torch.expm1(torch.tensor(init_eta))).item())  # inv-softplus
        theta_raw0 = math.log(init_theta / max(1e-6, (1.0 - init_theta)))        # inv-sigmoid
        self.eta_energy_raw = nn.Parameter(torch.tensor(eta_raw0))
        self.theta_frag_raw = nn.Parameter(torch.tensor(theta_raw0))

        # P0: outside FULL, freeze PRA knobs (strict orthogonality)
        if self.mode != "FULL":
            for p in [
                self.kappa_raw,
                self.pres_tau_logit,
                self.pres_temp_raw,
                self.abs_bias_mag_raw,
                self.eta_energy_raw,
                self.theta_frag_raw,
            ]:
                try:
                    p.requires_grad_(False)
                except Exception:
                    pass

    @staticmethod
    def compute_presence_proxy(mode: str, M_logits: torch.Tensor, probes: torch.Tensor | None):
        """Compute continuous presence proxy (Pc_prob + scalar p_present in [0,1])."""
        B = int(M_logits.shape[0])
        device = M_logits.device

        if (probes is not None) and torch.is_tensor(probes) and (probes.numel() > 0) and (probes.shape[1] >= 1) and bool(getattr(PRAAdapter, "_USE_PROBE_PRESENCE", False)):
            Pc = probes[:, 0:1]
            Pc_up = F.interpolate(Pc, size=M_logits.shape[2:], mode="bilinear", align_corners=False)
            Pc_prob = Pc_up if (Pc_up.min() >= 0.0 and Pc_up.max() <= 1.0) else torch.sigmoid(Pc_up)
        else:
            Pc_prob = torch.sigmoid(M_logits).detach()

        p_present = Pc_prob.flatten(2).amax(dim=-1).view(B).clamp(0.0, 1.0).to(dtype=torch.float32)
        return Pc_prob, p_present

    def forward(self, M_logits: torch.Tensor, Pc_prob: torch.Tensor, p_present: torch.Tensor):
        """Apply reflex refinement. Returns (M_fused_logits, reset_flags)."""
        B = int(M_logits.shape[0])
        device = M_logits.device

        if not self.enabled:
            return M_logits, torch.zeros(B, dtype=torch.bool, device=device)

        # Learned presence gate params (no legacy thresholds)
        tau = torch.sigmoid(self.pres_tau_logit).clamp(1e-4, 1.0 - 1e-4)
        temp = F.softplus(self.pres_temp_raw).clamp(min=1e-3)
        abs_bias = -F.softplus(self.abs_bias_mag_raw).clamp(min=0.0, max=20.0)

        # GFU (only boosts negative logits; protects SAM manifold)
        M_fused_logits = ReflexMechanisms.guided_fusion_unit(M_logits, Pc_prob, F.softplus(self.kappa_raw))

        # firewall thresholds
        eta_energy = F.softplus(self.eta_energy_raw).clamp(min=0.0)
        theta_frag = torch.sigmoid(self.theta_frag_raw).clamp(0.0, 1.0)
        flush_flags = ReflexMechanisms.causal_firewall(Pc_prob, eta_energy=eta_energy, theta_frag=theta_frag)
        if torch.is_tensor(flush_flags) and flush_flags.dim() != 1:
            flush_flags = flush_flags.view(B)

        # continuous suppression
        gate = torch.sigmoid((p_present - tau) / temp).clamp(0.0, 1.0)  # 1=present, 0=absent
        M_fused_logits = M_fused_logits + (1.0 - gate).view(B, 1, 1, 1) * abs_bias

        absent_flags = p_present < tau
        reset_flags = (flush_flags | absent_flags).to(dtype=torch.bool)

        return M_fused_logits, reset_flags
