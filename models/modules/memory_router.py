"""Memory routing & utility prediction (paper-grade module).

This module centralizes the *fully-learned* keyframe router used by RootSAM-Pro:
  - SoftWrite gate: write_weight in [0,1]
  - Dual-bank split: split_weight in [0,1]  (recent vs prompted)
  - InfoGain value head: per-bank utility in [0,1]^2

Design goals:
  - No hand-tuned thresholds / sweeps.
  - Deterministic ablation: it is a single module with explicit inputs/outputs.
  - Checkpoint-safe: does not mutate external state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bkmc import apply_delta_trust_ceiling


def mean_value(v: torch.Tensor | None) -> float:
    """Safe scalarization for priority decisions."""
    if v is None or (not torch.is_tensor(v)):
        return 0.0
    try:
        return float(v.detach().float().mean().item())
    except Exception:
        return 0.0


def max_ptr_sim(ptr_now: torch.Tensor | None, ptr_list: list, B: int, device) -> torch.Tensor:
    """Return max cosine similarity between current pointer and a list of stored pointers.

    Output: (B,) in [0,1]. If list empty or incompatible, returns zeros.
    """
    if ptr_now is None or not torch.is_tensor(ptr_now) or len(ptr_list) == 0:
        return torch.zeros(B, device=device, dtype=torch.float32)
    p0 = ptr_now
    if p0.dim() == 3:
        p0 = p0.flatten(1)
    elif p0.dim() == 2:
        p0 = p0
    else:
        p0 = p0.view(B, -1)
    sims = []
    for p in ptr_list:
        if p is None or (not torch.is_tensor(p)):
            continue
        p1 = p
        if p1.dim() == 3:
            p1 = p1.flatten(1)
        elif p1.dim() == 2:
            p1 = p1
        else:
            p1 = p1.view(B, -1)
        if p1.shape[-1] != p0.shape[-1]:
            continue
        sims.append(F.cosine_similarity(p0, p1, dim=-1).clamp(0.0, 1.0))
    if len(sims) == 0:
        return torch.zeros(B, device=device, dtype=torch.float32)
    return torch.stack(sims, dim=0).amax(dim=0)


def compute_uncertainty_features(logits_b1hw: torch.Tensor) -> torch.Tensor:
    """Return uncertainty feature vector (B,3): [entropy, boundary-entropy, gradient]."""
    p = torch.sigmoid(logits_b1hw.float()).clamp(1e-4, 1.0 - 1e-4)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    ent_mean = ent.flatten(1).mean(dim=1)

    dx = (p[..., 1:] - p[..., :-1]).abs()
    dy = (p[..., 1:, :] - p[..., :-1, :]).abs()
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    grad = (dx + dy).clamp(0.0, 1.0)
    grad_mean = grad.flatten(1).mean(dim=1)

    bnd_ent = (ent * grad).flatten(1).mean(dim=1)
    return torch.stack([ent_mean, bnd_ent, grad_mean], dim=1)


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Gradient reversal layer for primal-dual ascent (no new hyperparams)."""

    class _GR(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_in):
            ctx.scale = scale
            return x_in

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.scale * grad_output

    return _GR.apply(x)


@dataclass
class RouterOut:
    write_gate: torch.Tensor          # (B,)
    split_weight: torch.Tensor        # (B,)
    val_prob2: torch.Tensor           # (B,2)
    gate_loss: torch.Tensor           # scalar
    x_gate: torch.Tensor              # (B,11) float32


class MemoryRouter(nn.Module):
    """Fully-learned router + per-bank value head."""

    def __init__(self):
        super().__init__()

        self.key_in_norm = nn.LayerNorm(11)
        self.keyframe_mlp = nn.Sequential(
            nn.Linear(11, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.keyframe_mlp[-1].weight)
        nn.init.zeros_(self.keyframe_mlp[-1].bias)

        self.value_in_norm = nn.LayerNorm(11)
        self.value_mlp = nn.Sequential(
            nn.Linear(11, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.value_mlp[-1].weight)
        nn.init.zeros_(self.value_mlp[-1].bias)

        init_key_temp = 0.35
        key_temp_raw0 = float(torch.log(torch.expm1(torch.tensor(init_key_temp))).item())
        self.key_temp_raw = nn.Parameter(torch.tensor(key_temp_raw0))

        # router regularizer weights (learned): [bce, w_ent, s_ent, redund, value]
        self.key_log_vars = nn.Parameter(torch.zeros(5, dtype=torch.float32))

        # resource Lagrangian
        init_lambda = 0.1
        lambda_raw0 = float(torch.log(torch.expm1(torch.tensor(init_lambda))).item())
        self.resource_lambda_raw = nn.Parameter(torch.tensor(lambda_raw0))

    def forward(
        self,
        *,
        dt_days: torch.Tensor,
        sim: torch.Tensor,
        p_present: torch.Tensor,
        old_is_mem_empty: torch.Tensor,
        logits_b1hw: torch.Tensor,
        obj_ptr_decoder: torch.Tensor | None,
        recent_ptrs: list,
        prompt_ptrs: list,
        full_recent: bool,
        full_prompted: bool,
        reset_flags: torch.Tensor,
        mode: str,
        ptr_half_life_raw: torch.nn.Parameter,
    ) -> RouterOut:

        device = logits_b1hw.device
        B = int(logits_b1hw.shape[0])

        u_feat = compute_uncertainty_features(logits_b1hw)
        u_ent, u_bnd, u_grad = u_feat[:, 0], u_feat[:, 1], u_feat[:, 2]

        # adjust similarity using last pointer if compatible
        sim_adj = sim
        if (obj_ptr_decoder is not None) and (len(recent_ptrs) > 0):
            last_ptr = recent_ptrs[-1]
            if torch.is_tensor(last_ptr) and torch.is_tensor(obj_ptr_decoder):
                if last_ptr.shape[-1] == obj_ptr_decoder.shape[-1]:
                    sim_adj = F.cosine_similarity(obj_ptr_decoder.view(B, -1), last_ptr.view(B, -1), dim=-1).clamp(0.0, 1.0)

        max_sim_recent = max_ptr_sim(obj_ptr_decoder, recent_ptrs, B, device)
        max_sim_prompt = max_ptr_sim(obj_ptr_decoder, prompt_ptrs, B, device)
        novelty_recent = (1.0 - max_sim_recent).clamp(0.0, 1.0)
        novelty_prompt = (1.0 - max_sim_prompt).clamp(0.0, 1.0)

        full_recent_t = torch.full((B,), float(full_recent), device=device, dtype=torch.float32)
        full_prompt_t = torch.full((B,), float(full_prompted), device=device, dtype=torch.float32)

        x_gate = torch.stack(
            [
                dt_days.float(),
                (1.0 - sim_adj).clamp(0.0, 1.0),
                p_present.clamp(0.0, 1.0),
                old_is_mem_empty.float(),
                u_ent,
                u_bnd,
                u_grad,
                novelty_recent,
                novelty_prompt,
                full_recent_t,
                full_prompt_t,
            ],
            dim=1,
        ).to(dtype=torch.float32)

        x_gate_n = self.key_in_norm(x_gate)
        logits2 = self.keyframe_mlp(x_gate_n)
        write_logit = logits2[:, 0].contiguous().view(B)
        split_logit = logits2[:, 1].contiguous().view(B)

        write_prob = torch.sigmoid(write_logit)
        split_prob = torch.sigmoid(split_logit)

        is_training = self.training and torch.is_grad_enabled()
        if is_training:
            u1 = torch.rand_like(write_logit).clamp(1e-6, 1.0 - 1e-6)
            u2 = torch.rand_like(split_logit).clamp(1e-6, 1.0 - 1e-6)
            g1 = -torch.log(-torch.log(u1))
            g2 = -torch.log(-torch.log(u2))
            temp = F.softplus(self.key_temp_raw).clamp(min=1e-3)
            write_weight = torch.sigmoid((write_logit + g1) / temp)
            split_weight = torch.sigmoid((split_logit + g2) / temp)
        else:
            write_weight = write_prob
            split_weight = split_prob

        val_logits2 = self.value_mlp(self.value_in_norm(x_gate))
        val_prob2 = torch.sigmoid(val_logits2).clamp(0.0, 1.0)

        write_weight = (write_weight * (~reset_flags).float()).clamp(0.0, 1.0)

        if mode in ["SRD_KMR", "FULL"]:
            write_weight, write_prob, _cap = apply_delta_trust_ceiling(
                write_weight=write_weight,
                write_prob=write_prob,
                dt_days=dt_days.float(),
                half_life_raw=ptr_half_life_raw,
            )

        split_weight = split_weight.clamp(0.0, 1.0)

        # --- regularizers ---
        teacher_logit = x_gate.sum(dim=1)
        teacher_prob = torch.sigmoid(teacher_logit).clamp(0.0, 1.0)
        teacher_prob = teacher_prob * p_present.clamp(0.0, 1.0) * (~reset_flags).float()
        loss_gate_bce = F.binary_cross_entropy_with_logits(write_logit, teacher_prob.detach(), reduction="mean")

        loss_write_ent = -(
            write_weight * torch.log(write_weight + 1e-6)
            + (1.0 - write_weight) * torch.log(1.0 - write_weight + 1e-6)
        ).mean()
        loss_split_ent = -(
            split_weight * torch.log(split_weight + 1e-6)
            + (1.0 - split_weight) * torch.log(1.0 - split_weight + 1e-6)
        ).mean()

        recent_write = (write_weight * (1.0 - split_weight)).mean()
        prompt_write = (write_weight * split_weight).mean()
        loss_redund = (recent_write * max_sim_recent.mean() + prompt_write * max_sim_prompt.mean())

        u_mean = ((u_ent + u_bnd + u_grad) / 3.0).clamp(0.0, 1.0)
        teacher_recent_v = (novelty_recent * u_mean * p_present * (~reset_flags).float()).clamp(0.0, 1.0)
        teacher_prompt_v = (novelty_prompt * u_mean * p_present * (~reset_flags).float()).clamp(0.0, 1.0)
        teacher_v = torch.stack([teacher_recent_v, teacher_prompt_v], dim=1)
        loss_value = F.mse_loss(val_prob2, teacher_v.detach())

        raw_losses = [loss_gate_bce, loss_write_ent, loss_split_ent, loss_redund, loss_value]
        gate_loss = logits_b1hw.new_zeros(())
        for i, lv in enumerate(raw_losses):
            precision = torch.exp(-self.key_log_vars[i])
            gate_loss = gate_loss + precision * lv + 0.5 * self.key_log_vars[i].clamp(-5.0, 5.0)

        eviction_cost = recent_write * float(full_recent) + prompt_write * float(full_prompted)
        redundancy_full = (
            recent_write * float(full_recent) * max_sim_recent.mean()
            + prompt_write * float(full_prompted) * max_sim_prompt.mean()
        )
        resource_cost = eviction_cost + redundancy_full
        lambda_evict = F.softplus(grad_reverse(self.resource_lambda_raw)).clamp(0.0, 50.0)
        gate_loss = gate_loss + lambda_evict * resource_cost

        if not is_training:
            gate_loss = gate_loss.detach() * 0.0

        return RouterOut(
            write_gate=write_weight,
            split_weight=split_weight,
            val_prob2=val_prob2,
            gate_loss=gate_loss,
            x_gate=x_gate,
        )
