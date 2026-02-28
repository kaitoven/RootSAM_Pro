import math

import torch
import torch.nn as nn

from ..modules.ppfsrd_core import FourierWashingUnit


class SRDAdapter(nn.Module):
    """Spectro-Rank Decoupled Adapter (SRD) — PP-FSRD++ (phase-preserving Fourier).

    This SRD uses a phase-preserving Fourier washer in the low-rank bottleneck to
    attenuate background energy while keeping geometric phase intact.

    Backward compatibility:
      - accepts the legacy keyword `use_wavelet` as an alias of `use_fourier`.

    Safety features retained:
      - 3D [B,L,D] or 4D [B,H,W,D] inputs
      - supports global tokens (prefix) in 3D mode
      - contiguous memory before reshape
      - W_up is zero-init but gamma is non-zero to avoid gradient deadlock
    """

    def __init__(self, d_model: int = 1024, m_rank: int = 4, use_fourier: bool = True, **kwargs):
        super().__init__()
        self.d_model = int(d_model)
        self.m_rank = int(m_rank)

        # Legacy alias support: SRDAdapter(..., use_wavelet=...) still works.
        if "use_wavelet" in kwargs:
            use_fourier = bool(kwargs["use_wavelet"])

        self.use_fourier = bool(use_fourier)

        self.W_down = nn.Linear(self.d_model, self.m_rank, bias=False)
        if self.use_fourier:
            self.fourier_washer = FourierWashingUnit(self.m_rank)

        self.act = nn.GELU()
        self.W_up = nn.Linear(self.m_rank, self.d_model, bias=False)

        # Do NOT zero both W_up and gamma.
        self.gamma = nn.Parameter(torch.tensor(1.0))

        nn.init.kaiming_uniform_(self.W_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_up.weight)

    def forward(self, x: torch.Tensor, hw_shape=None) -> torch.Tensor:
        orig_dim = x.dim()
        orig_shape = x.shape

        if orig_dim == 4:
            B, H, W, D = x.shape
            x_flat = x.view(B, H * W, D)
            hw_shape = (H, W)
            num_globals = 0
        elif orig_dim == 3:
            B, L, D = x.shape
            x_flat = x
            if hw_shape is None:
                s = math.sqrt(L)
                if float(s).is_integer():
                    hw_shape = (int(s), int(s))
                else:
                    # fallback: assume 1 global token
                    s2 = math.sqrt(max(1, L - 1))
                    hw_shape = (int(s2), int(s2))
            H, W = hw_shape
            num_globals = L - (H * W)
            if num_globals < 0:
                raise ValueError(f"Invalid hw_shape={hw_shape} for L={L}")
        else:
            raise ValueError(f"SRDAdapter expects 3D or 4D tensors, got {orig_dim}D")

        z = self.W_down(x_flat)  # (B, L, m_rank)

        if self.use_fourier:
            globals_z = z[:, :num_globals, :] if num_globals > 0 else None
            spatial_z = z[:, num_globals:, :]

            z_2d = spatial_z.transpose(1, 2).contiguous().view(B, self.m_rank, H, W)
            z_washed = self.fourier_washer(z_2d)
            z_spatial_out = z_washed.flatten(2).transpose(1, 2)
            z = torch.cat([globals_z, z_spatial_out], dim=1) if globals_z is not None else z_spatial_out

        z_active = self.act(z)
        out_flat = self.gamma * self.W_up(z_active)

        if orig_dim == 4:
            return out_flat.view(orig_shape)
        return out_flat
