import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(x_0, t)  # (not used: legacy)
    noise = sde.marginal_std(e, t)
    x_t = x_mean + noise
    score = -noise
    output = model(x_t, t)
    loss = (output - score).square().sum(dim=(1,2,3)).mean(dim=0)
    return loss

def _trapz_weights(x: torch.Tensor) -> torch.Tensor:
    """Trapezoid weights for (B,N) coordinates."""
    assert x.dim() == 2, "x must be (B, N)"
    dx = (x[:, 1:] - x[:, :-1]).abs()
    w = torch.zeros_like(x)
    if x.size(1) > 2:
        w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    w[:, 0]  = 0.5 * dx[:, 0]
    w[:, -1] = 0.5 * dx[:, -1]
    return w

# ─────────────────────────────────────────────────────────────
# NUDFT utility (only 1 forward call in loss)
# ─────────────────────────────────────────────────────────────
def _iter_kappas_with_meta(model):
    """
    Iterate through all spectral blocks in the model and return (kappa, split_fracs).
    - κ is detached from the loss to avoid backpropagation.
    - If there are no spectral blocks, return an empty list.
    """
    mm = model.module if hasattr(model, "module") else model
    res = []
    if hasattr(mm, "spectral_blocks") and len(mm.spectral_blocks) > 0:
        for blk in mm.spectral_blocks:
            if hasattr(blk, "kappa_full"):
                kappa = blk.kappa_full().detach()
                split_fracs = getattr(blk, "band_split_fracs", (0.4, 0.8))
                res.append((kappa, split_fracs))
    return res

def _forward_nudft_real(r: torch.Tensor, z: torch.Tensor, kappa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    r : (B,N) real residual
    z : (B,N) phase coordinate (here z = π * x_norm)
    kappa : (M,) non-uniform frequency
    return: (Fr, Fi) each (B,M) — NUDFT(real/imaginary)
    """
    B, N = r.shape
    M = int(kappa.numel())
    phase = z.unsqueeze(-1) * kappa.view(1, 1, M)      # (B,N,M)
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    x = r.unsqueeze(1)                                 # (B,1,N)
    # (B,1,N) @ (B,N,M) -> (B,1,M)
    Fr = torch.bmm(x, cos).squeeze(1) / max(N, 1)      # (B,M)
    Fi = torch.bmm(x, -sin).squeeze(1) / max(N, 1)     # (B,M)
    return Fr, Fi

def _band_weight_schedule(
    kappa: torch.Tensor,
    *,
    split_fracs: Tuple[float, float] = (0.4, 0.8),
    global_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    |kappa| by quantiles to divide into low/mid/high bands and create band-specific schedule weights.
    Early (low-heavy) → Late (high-heavy) linear interpolation. Normalize to final mean = 1.
    return: w_k (M,)
    """
    k_abs = kappa.abs()
    q1, q2 = split_fracs
    try:
        th1 = torch.quantile(k_abs, q1)
        th2 = torch.quantile(k_abs, q2)
    except Exception:
        k_sorted, _ = torch.sort(k_abs)
        i1 = int((len(k_sorted) - 1) * q1)
        i2 = int((len(k_sorted) - 1) * q2)
        th1, th2 = k_sorted[i1], k_sorted[i2]

    low  = k_abs <= th1
    mid  = (k_abs > th1) & (k_abs <= th2)
    high = k_abs > th2

    if (global_step is None) or (max_steps is None) or max_steps <= 0:
        s = 0.0
    else:
        s = float(global_step) / float(max_steps)
        s = max(0.0, min(1.0, s))

    # Start/end band weights (adjustable if needed)
    low_start,  mid_start,  high_start  = 1.00, 0.30, 0.10
    low_final,  mid_final,  high_final  = 0.20, 0.80, 1.00

    wL = (1.0 - s) * low_start  + s * low_final
    wM = (1.0 - s) * mid_start  + s * mid_final
    wH = (1.0 - s) * high_start + s * high_final

    w = torch.empty_like(kappa)
    w[low]  = wL
    w[mid]  = wM
    w[high] = wH

    # Normalize to mean 1 (stable scale throughout training)
    w = w * (w.numel() / (w.sum() + 1e-12))
    return w

def hilbert_loss_fn(
    model,
    sde,
    x_0: torch.Tensor,                # (B,N) normalized target
    t: torch.Tensor,                  # (B,)
    e: torch.Tensor,                  # (B,N) Hilbert noise
    x_coord: torch.Tensor,            # (B,N) normalized coordinates [-1,1]
    *,
    global_step: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    (1) Coordinate-integrated MSE (existing Hilbert loss)
    (2) NUDFT-based spectral-weighted residual loss (optional)
        L_spec = Σ_k w_k |r~(k)|^2,  r = (output - target)
    """
    # ----- 1) Standard (coordinate-weighted) score loss -----
    x_mean = sde.diffusion_coeff(t)               # ᾱ(t)
    noise  = sde.marginal_std(t)                  # σ(t)
    x_t    = x_0 * x_mean[:, None] + e * noise[:, None]
    target = -e

    model_input = torch.cat([x_t.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)
    output = model(model_input, t.float())        # (B,N) — score(x_t)

    # trapezoid weighted average to preserve scale
    w_trapz = _trapz_weights(x_coord)
    w_trapz = w_trapz / (w_trapz.sum(dim=1, keepdim=True) + 1e-12)
    data_loss = ((output - target).square() * w_trapz).sum(dim=1).mean(dim=0)

    return data_loss

