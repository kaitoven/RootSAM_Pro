import torch
import torch.nn as nn
import torch.nn.functional as F

class BHFI(nn.Module):
    """
    Boundary-guided High-Frequency Injection (BHFI)

    Not a U-Net style skip. This is a light, gated residual on mask logits:
      - Predicts a delta-logit from sanitized HR-FPN features (feat_s0/feat_s1)
      - Injects ONLY near predicted boundaries: b = 4*p*(1-p)
      - Zero-init heads + tiny beta gate => starts as identity (safe on pretrained SAM2 manifold)

    Expected inputs:
      - dec_out: mask decoder output, first element should be masks logits (B,N,H,W)
      - feat_s0: (B,32,H0,W0)
      - feat_s1: (B,64,H1,W1)

    Returns:
      - dec_out with refined masks logits.
    """
    def __init__(self, c0: int = 32, c1: int = 64, init_beta_raw: float = -6.0):
        super().__init__()
        self.act = nn.GELU()

        self.dw0 = nn.Conv2d(c0, c0, kernel_size=3, padding=1, groups=c0, bias=False)
        self.pw0 = nn.Conv2d(c0, 1, kernel_size=1, bias=True)

        self.dw1 = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)
        self.pw1 = nn.Conv2d(c1, 1, kernel_size=1, bias=True)

        self.beta_raw = nn.Parameter(torch.tensor(float(init_beta_raw)))

        # Safe init: delta heads are zero; depthwise conv small random
        nn.init.normal_(self.dw0.weight, std=0.02)
        nn.init.normal_(self.dw1.weight, std=0.02)
        nn.init.zeros_(self.pw0.weight); nn.init.zeros_(self.pw0.bias)
        nn.init.zeros_(self.pw1.weight); nn.init.zeros_(self.pw1.bias)

    @torch.no_grad()
    def beta(self) -> float:
        return float(torch.sigmoid(self.beta_raw).clamp(0.0, 1.0).cpu().item())

    def forward(self, dec_out, feat_s0: torch.Tensor, feat_s1: torch.Tensor):
        masks = dec_out[0] if isinstance(dec_out, (tuple, list)) else dec_out
        if not torch.is_tensor(masks):
            return dec_out

        # masks: (B,N,H,W)
        H, W = int(masks.shape[-2]), int(masks.shape[-1])

        d0 = self.pw0(self.act(self.dw0(feat_s0)))
        d1 = self.pw1(self.act(self.dw1(feat_s1)))
        if d1.shape[-2:] != d0.shape[-2:]:
            d1 = F.interpolate(d1, size=d0.shape[-2:], mode="bilinear", align_corners=False)
        delta = d0 + d1
        if delta.shape[-2:] != (H, W):
            delta = F.interpolate(delta, size=(H, W), mode="bilinear", align_corners=False)

        # boundary gate from current prediction (no thresholds)
        p = torch.sigmoid(masks)
        b_gate = (4.0 * p * (1.0 - p)).clamp(0.0, 1.0)  # (B,N,H,W)

        beta = torch.sigmoid(self.beta_raw).to(device=masks.device, dtype=masks.dtype).clamp(0.0, 1.0)
        masks_ref = masks + beta * b_gate * delta.to(dtype=masks.dtype)

        if isinstance(dec_out, tuple):
            return (masks_ref,) + tuple(dec_out[1:])
        if isinstance(dec_out, list):
            return [masks_ref] + list(dec_out[1:])
        return masks_ref
