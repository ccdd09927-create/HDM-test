import torch
import torchode as to
from typing import Optional
from functions.sde import VPSDE1D
# -----------------------------------------------------------------------------
# Resolution-invariant RMS clip: approximate continuous-domain L2 average by trapezoidal integration
# -----------------------------------------------------------------------------
def _trapz_weights(x_coord: torch.Tensor) -> torch.Tensor:
    """
    x_coord: (B, N)
    To handle non-uniform coordinates, create trapezoidal weights based on sorted coordinates for each batch.
    """
    xs, _ = torch.sort(x_coord, dim=1)           # (B, N)
    if xs.size(1) == 1:
        return torch.ones_like(xs)

    dx = xs[:, 1:] - xs[:, :-1]                  # (B, N-1)
    w = torch.zeros_like(xs)                     # (B, N)
    w[:, 0] = 0.5 * dx[:, 0]
    w[:, -1] = 0.5 * dx[:, -1]
    if xs.size(1) > 2:
        w[:, 1:-1] = 0.5 * (dx[:, 1:] + dx[:, :-1])
    return w.clamp_min(0.0)


def rms_clip_resolution_free(
    x: torch.Tensor, x_coord: torch.Tensor, thr: float
) -> torch.Tensor:
    """
    x: (B, N)
    x_coord: (B, N)
    thr: RMS upper bound. sqrt( (1/L) ∫ x(z)^2 dz ) <= thr 

    Instead of resolution-dependent vector L2 norm, use continuous-domain RMS to ensure amplitude stability across res_free_points variations.
    """
    if not torch.isfinite(torch.tensor(thr)) or thr <= 0:
        return x

    w = _trapz_weights(x_coord)                                   # (B, N)
    L = w.sum(dim=1, keepdim=True).clamp_min(1e-12)               # (B, 1)
    rms = torch.sqrt((w * x.pow(2)).sum(dim=1, keepdim=True) / L) # (B, 1)

    mask = (rms > thr).squeeze(1)                                 # (B,)
    if mask.any():
        x_scaled = x[mask] * (thr / rms[mask])
        x = x.clone()
        x[mask] = x_scaled
    return x

# -----------------------------------------------------------------------------
# Tsitouras 5(4) Probability-flow ODE sampler (resolution-free)
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_probability_flow_ode(
    model,
    sde: VPSDE1D,
    *,
    x_t0: torch.Tensor,                 # (B, N) initial state: x_T
    x_coord: torch.Tensor,              # (B, N) coordinates
    inference_steps: int = 500,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    device: str = "cuda",
    # Optional: resolution-invariant RMS clip for divergence prevention (default disabled)
    enable_rms_clip: bool = False,
    rms_clip_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Resolution-free probability-flow ODE sampler ― Tsitouras 5(4) + IntegralController.

    - Use variable transformation s = T - t (monotonically increasing) for reverse-time integration.
    - Assuming the model output approximates the predicted noise -e, transform score(t, x) = model_out / sigma(t).
    - The probability-flow ODE (VPSDE): dx = [-1/2 * beta(t) * x - 1/2 * beta(t) * score(t, x)] dt, therefore dz/ds = -f(t, z).
    - Removed the resolution-dependent amplitude-reduction post-processing (L2 clipping).
    - Optionally provide resolution-invariant RMS clip for divergence prevention.
    """
    model.eval()

    # Device/shape alignment
    x = x_t0.to(device)
    x_coord = x_coord.to(device)
    batch, dim = x.shape

    T = sde.T
    eps = sde.eps

    # -----------------------------
    # 1) Reverse-time ODE: dz/ds = -f(t, z)
    # -----------------------------
    def reverse_f(s, y):
        # s \in [eps, T]  ->  t = T - s + eps \in [T, eps]
        t = T - s + eps
        t_vec = t.expand(batch)                           # (B,)

        # Model input: (B, 2, N)  [signal, coord]
        model_input = torch.cat([y.unsqueeze(1), x_coord.unsqueeze(1)], dim=1)

        # Model output approximates -e_hat. score = (-e_hat) / sigma(t) = model_out / sigma(t)
        model_out = model(model_input, t_vec)             # (B, N)
        sigma_t = sde.marginal_std(t_vec)                 # (B,)
        score = model_out / sigma_t[:, None]              # (B, N)

        beta_t = sde.beta(t_vec)                          # (B,)
        # f(t, y) = -1/2 * beta * (y + score)
        forward_drift = -0.5 * beta_t[:, None] * (y + score)

        # dz/ds = - f(t, z)  (s = T - t)
        return -forward_drift

    # -----------------------------
    # 2) ODE solver configuration
    # -----------------------------
    term = to.ODETerm(reverse_f)
    step = to.Tsit5(term=term)
    ctrl = to.IntegralController(atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step, ctrl)

    # -----------------------------
    # 3) Integration interval and IVP definition
    # -----------------------------
    s_eval = torch.linspace(eps, T, inference_steps + 1, device=device).expand(batch, -1)
    problem = to.InitialValueProblem(y0=x, t_eval=s_eval)

    # -----------------------------
    # 4) Integration execution
    # -----------------------------
    sol = solver.solve(problem)
    x0 = sol.ys[:, -1]                                     # (B, N)

    # -----------------------------
    # 5) (Optional) Resolution-invariant RMS clip
    # -----------------------------
    if enable_rms_clip and (rms_clip_threshold is not None):
        x0 = rms_clip_resolution_free(x0, x_coord, float(rms_clip_threshold))

    # Do not apply resolution-dependent global L2 clipping.
    return x0