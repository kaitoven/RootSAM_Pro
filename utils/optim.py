import torch
from typing import Iterable, Tuple, List, Dict, Any


def build_adamw_param_groups(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Parameter-wise weight decay for AdamW.

    - Apply weight decay ONLY to matrix-like parameters (ndim >= 2).
    - Do NOT decay: biases, normalization params, scalar gates, and dual/log-variance variables.
    """
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for name, p in named_params:
        if p is None or not getattr(p, "requires_grad", False):
            continue
        n = name.lower()
        is_bias = n.endswith(".bias") or ".bias" in n or "bias" in n
        is_norm = ("norm" in n) or ("layernorm" in n) or ("batchnorm" in n) or ("groupnorm" in n) or (".ln" in n)
        is_dual_like = ("soil_lambda" in n) or ("log_vars" in n)

        if p.ndim <= 1 or is_bias or is_norm or is_dual_like:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_adamw_param_groups_dual_lr(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    weight_decay: float,
    base_lr: float,
    dual_lr_mult: float = 10.0,
) -> List[Dict[str, Any]]:
    """Parameter-wise weight decay + separate dual-variable LR group.

    - weight_decay applies only to matrix-like weights (ndim>=2), excluding bias/norm.
    - dual variables (soil_lambda_raw / soil_lambda) are put into their own group:
        lr = base_lr * dual_lr_mult, weight_decay = 0

    This is a standard primal-dual optimization trick.
    """
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    dual: List[torch.nn.Parameter] = []

    for name, p in named_params:
        if p is None or not getattr(p, "requires_grad", False):
            continue

        n = name.lower()
        is_bias = n.endswith(".bias") or ".bias" in n or "bias" in n
        is_norm = ("norm" in n) or ("layernorm" in n) or ("batchnorm" in n) or ("groupnorm" in n) or (".ln" in n)
        is_dual = ("soil_lambda_raw" in n) or ("soil_lambda" in n)

        if is_dual:
            dual.append(p)
            continue

        if p.ndim <= 1 or is_bias or is_norm or ("log_vars" in n):
            no_decay.append(p)
        else:
            decay.append(p)

    groups: List[Dict[str, Any]] = []
    if len(decay) > 0:
        groups.append({"params": decay, "weight_decay": float(weight_decay), "lr": float(base_lr)})
    if len(no_decay) > 0:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr": float(base_lr)})
    if len(dual) > 0:
        groups.append({"params": dual, "weight_decay": 0.0, "lr": float(base_lr) * float(dual_lr_mult)})

    return groups
