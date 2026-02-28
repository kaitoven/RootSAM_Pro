import math
import torch
import torch.nn.functional as F

def apply_delta_trust_ceiling(
    write_weight: torch.Tensor,
    write_prob: torch.Tensor,
    dt_days: torch.Tensor,
    half_life_raw: torch.Tensor,
):
    """
    Δt-aware trust ceiling (no heuristics):
      - dt only sets an *upper bound* on how much we can trust/write memory.
      - avoids large-Δt hallucinated writes, especially on PRMI with frequent flips.
      - reuses Bio-KES half-life (no extra cfg hyperparams).

    Args:
      write_weight: (B,1) or (B,) in [0,1]
      write_prob:   (B,1) or (B,) in [0,1] (pre-sigmoid prob if you keep both)
      dt_days:      (B,) in days
      half_life_raw: learnable parameter tensor (any shape), transformed by softplus to days

    Returns:
      write_weight_cap, write_prob_cap, cap (B,)
    """
    # robust scalar half-life in days
    hl = F.softplus(half_life_raw).mean().clamp(min=1.0, max=10000.0)  # days
    cap = torch.exp((-math.log(2.0) * dt_days.float() / hl)).clamp(0.0, 1.0)  # (B,)
    cap = cap.to(device=write_weight.device, dtype=write_weight.dtype)

    write_weight_cap = (write_weight * cap).clamp(0.0, 1.0)
    write_prob_cap = (write_prob * cap).clamp(0.0, 1.0)
    return write_weight_cap, write_prob_cap, cap
