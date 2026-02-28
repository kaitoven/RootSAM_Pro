import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaAdapter(nn.Module):
    """A tiny PEFT adapter baseline (for VANILLA ablation).

    This is intentionally minimal: a low-rank bottleneck with zero-init up-proj.
    """

    def __init__(self, d_model: int = 1024, m_rank: int = 4):
        super().__init__()
        self.down = nn.Linear(d_model, m_rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(m_rank, d_model, bias=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, hw_shape=None):
        # x: (B,L,C) or (B,H,W,C)
        orig_dim = x.dim()
        if orig_dim == 4:
            B, H, W, C = x.shape
            x_flat = x.reshape(B, H * W, C)
            z = self.act(self.down(x_flat))
            out = self.gamma * self.up(z)
            return out.reshape(B, H, W, C)
        z = self.act(self.down(x))
        return self.gamma * self.up(z)


class VanillaKMRAdapter(nn.Module):
    """A lightweight KMR baseline (for VANILLA/SRD_ONLY ablation).

    Upgraded for PRMI: tubular cross smoothing (anisotropic low-pass) to favor
    long rectilinear continuity and suppress point-like soil highlights.

    No thresholds, no extra cfg hyperparams.
    """

    def __init__(self, d_model: int = 256, m_attn: int = 32):
        super().__init__()
        self.down = nn.Conv2d(d_model, m_attn, kernel_size=1, bias=False)
        self.act = nn.GELU()

        # Tubular cross smoothing strength (per-channel), alpha = sigmoid(raw) in (0,1)
        # init raw=-3 => alpha≈0.047 (gentle, near-identity)
        self.tube_gate_raw = nn.Parameter(torch.full((1, m_attn, 1, 1), -3.0))

        self.up = nn.Conv2d(m_attn, d_model, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, F_track: torch.Tensor, delta_t: torch.Tensor):
        # F_track: (B,256,H,W)
        z = self.act(self.down(F_track))

        a = torch.sigmoid(self.tube_gate_raw).to(device=z.device, dtype=z.dtype)
        z_tube = (
            F.avg_pool2d(z, kernel_size=(1, 9), stride=1, padding=(0, 4)) +
            F.avg_pool2d(z, kernel_size=(9, 1), stride=1, padding=(4, 0))
        )
        z = (1.0 - a) * z + a * self.act(z_tube)

        out = F_track + self.gamma * self.up(z)
        B, _, H, W = out.shape
        dummy_probes = torch.full((B, 3, H, W), 0.5, device=out.device, dtype=out.dtype)
        dummy_F_kin = torch.ones((B, 1, 1, 1), device=out.device, dtype=out.dtype)
        return out, dummy_probes, dummy_F_kin


class SRDSafeWrapper(nn.Module):
    """Wrap SAM2 MLP so adapter is always added safely."""

    def __init__(self, orig_mlp: nn.Module, adapter: nn.Module | None):
        super().__init__()
        self.orig_mlp = orig_mlp
        self.adapter = adapter

    def forward(self, x, *args, **kwargs):
        if self.adapter is None:
            return self.orig_mlp(x, *args, **kwargs)
        hw_shape = kwargs.get("hw_shape", None)
        return self.orig_mlp(x, *args, **kwargs) + self.adapter(x, hw_shape=hw_shape)
