"""PP-FSRD++ core: Phase-Preserving Fourier washer with learnable polar spectral bands.

This file is the single source of truth for RootSAM-Pro's frequency-domain washing.

Design goals (RootSAM-Pro):
  - Preserve *phase* exactly (geometric topology / skeleton location)
  - Adapt only *amplitude* (soil texture energy / glitter noise)
  - No stride-2 decimation; keep full spatial resolution
  - AMP-safe: FFT in fp32, cast back to original dtype
  - Identity at init: mixer is dirac, band weights are zeros, gamma is zero

The module is used by SRDAdapter (Semantic-SRD) and can be reused elsewhere.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierWashingUnit(nn.Module):
    """Phase-Preserving Fourier Washer (PP-FSRD++).

    Given x in spatial domain, perform:
      rfft2(x) -> (amp, phase)
      amp <- amp + gamma * amp_mixer(amp) * exp(log_scale(r,theta))
      phase is kept unchanged
      irfft2(polar(amp, phase))

    Polar bands:
      - radial bases over r in [0,1]
      - orientation bases over theta in [0, pi/2] using atan2(|ky|, kx)
    """

    def __init__(self, channels: int, num_radial_bands: int = 6, num_orient_bands: int = 6):
        super().__init__()
        self.channels = int(channels)
        self.num_radial_bands = int(num_radial_bands)
        self.num_orient_bands = int(num_orient_bands)

        # Cross-channel amplitude mixer (1x1 in frequency domain).
        self.amp_mixer = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        nn.init.dirac_(self.amp_mixer.weight)

        # Learnable polar weights (C, Kr, Ko), init zeros => band_scale==1.
        self.polar_band_w = nn.Parameter(torch.zeros(self.channels, self.num_radial_bands, self.num_orient_bands))

        # Residual gate (zero-init => identity mapping at start).
        self.gamma = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

        # CPU caches for bases keyed by (H, W, K).
        self._radial_basis_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self._orient_basis_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        orig_dtype = x.dtype
        x_f = x.float()  # AMP-safe

        fft_x = torch.fft.rfft2(x_f, norm="ortho")
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        R = self._get_radial_basis(x.shape[-2], x.shape[-1], amp.device, amp.dtype)  # (Kr,H,Wf)
        A = self._get_orient_basis(x.shape[-2], x.shape[-1], amp.device, amp.dtype)  # (Ko,H,Wf)

        w = self.polar_band_w.to(dtype=amp.dtype, device=amp.device)  # (C,Kr,Ko)

        # Separable polar log-scale: (C,H,Wf)
        T = torch.einsum("crk,khw->crhw", w, A)
        log_scale = torch.einsum("crhw,rhw->chw", T, R)
        band_scale = torch.exp(log_scale.clamp(-2.0, 2.0)).unsqueeze(0)  # (1,C,H,Wf)

        amp_delta = self.amp_mixer(amp) * band_scale
        amp_out = F.relu(amp + self.gamma * amp_delta)

        fft_out = torch.polar(amp_out, phase)
        x_out = torch.fft.irfft2(fft_out, s=(x.shape[-2], x.shape[-1]), norm="ortho")
        return x_out.to(orig_dtype)

    @torch.no_grad()
    def _get_radial_basis(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Smooth radial bases R in [0,1], shape (Kr, H, Wf)."""
        key = (int(H), int(W), int(self.num_radial_bands))
        cached = self._radial_basis_cache.get(key, None)
        if cached is not None:
            return cached.to(dtype=dtype, device=device)

        ky = torch.fft.fftfreq(H, d=1.0, device=device).view(H, 1)
        kx = torch.fft.rfftfreq(W, d=1.0, device=device).view(1, W // 2 + 1)
        r = torch.sqrt(ky * ky + kx * kx)
        r = r / (r.max().clamp(min=1e-6))

        Kr = self.num_radial_bands
        centers = torch.linspace(0.0, 1.0, Kr, device=device).view(Kr, 1, 1)
        width = 1.0 / max(1, (Kr - 1))
        R = torch.relu(1.0 - torch.abs(r.unsqueeze(0) - centers) / (width + 1e-6))

        self._radial_basis_cache[key] = R.detach().cpu()
        return R.to(dtype=dtype, device=device)

    @torch.no_grad()
    def _get_orient_basis(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Smooth orientation bases A over theta in [0, pi/2], shape (Ko, H, Wf).

        Use theta = atan2(|ky|, kx) to be direction-invariant and rFFT compatible.
        """
        key = (int(H), int(W), int(self.num_orient_bands))
        cached = self._orient_basis_cache.get(key, None)
        if cached is not None:
            return cached.to(dtype=dtype, device=device)

        ky = torch.fft.fftfreq(H, d=1.0, device=device).view(H, 1)
        kx = torch.fft.rfftfreq(W, d=1.0, device=device).view(1, W // 2 + 1)
        theta = torch.atan2(ky.abs(), kx.clamp(min=1e-6))
        theta = theta / (0.5 * math.pi)  # normalize to [0,1]

        Ko = self.num_orient_bands
        centers = torch.linspace(0.0, 1.0, Ko, device=device).view(Ko, 1, 1)
        width = 1.0 / max(1, (Ko - 1))
        A = torch.relu(1.0 - torch.abs(theta.unsqueeze(0) - centers) / (width + 1e-6))

        self._orient_basis_cache[key] = A.detach().cpu()
        return A.to(dtype=dtype, device=device)
