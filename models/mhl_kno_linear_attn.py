import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import (
    get_timestep_embedding,
    Lifting,
    Projection,
    default_init,
)
from .mlp import MLP, skip_connection
from .mhl_kno import TimeDependentRFFKernel1D, _trapz_weights_1d
from .chebyshev_mhl_kno import TimeDependentChebyshevKernel1D


class LinearAttentionKNO1D(nn.Module):
    r"""
    Multi-head Linear-KNO layer in explicit linear-attention (Q,K,V) form.

    From Spectraformer perspective,
      - Use multiple RF families ϕ^{(m)}(x) simultaneously
      - Maintain overall feature budget D_k similar to baseline (RFF-only)
      - Unified RF layer that only varies inductive bias in kernel space.

    rf_backend:
      - "rff"       : Original RFF-only (same as MHLKNO)
      - "chebyshev" : Chebyshev/Taylor-only
      - "hybrid"    : Use RFF + Chebyshev within a single layer, sharing feature budget
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 32,
        num_basis: int = 16,          
        temb_dim: int = 256,
        bandwidth_init: float = 1.0,
        measure_scale: float = 1.0,
        *,
        rf_backend: str = "rff",       # "rff" | "chebyshev" | "hybrid"
        # unified RF budget 
        rf_total_basis: int | None = None,  # Total budget (Default: num_basis)
        rf_rff_frac: float = 0.5,           # RFF ratio in hybrid (0~1)
        # Common kernel hyperparameters
        kernel_type: str = "rbf",
        kernel_Q: int = 1,
        kernel_hidden: int = 64,
        enable_kernel_time_cond: bool = True,
        kernel_time_hidden: int = 64,
        hyp_len_init=None,
        rff_type: str = "ghq",
        bias: bool = True,
        enable_spatial_cond: bool = False,
        spatial_hidden: int = 64,
        # Chebyshev-only hyperparameters (default degree for Cheb-only mode)
        taylor_degree: int = 10,
    ):
        super().__init__()

        self.num_heads = int(num_heads)
        self.measure_scale = float(measure_scale)

        if in_channels % self.num_heads != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim = in_channels // self.num_heads  # D_h

        # baseline: RFF-only일 때 num_basis = cfg.num_kernel_basis
        self.num_basis_cfg = int(num_basis)

        # Total feature budget: use baseline num_basis as is (can override with rf_total_basis)
        if rf_total_basis is None:
            rf_total_basis = self.num_basis_cfg
        self.rf_total_basis = int(rf_total_basis)
        self.rf_backend = str(rf_backend).lower()

        # Determine which backend to use
        if self.rf_backend == "rff":
            self.use_rff = True
            self.use_cheb = False
        elif self.rf_backend == "chebyshev":
            self.use_rff = False
            self.use_cheb = True
        elif self.rf_backend in ("hybrid", "unified"):
            self.use_rff = True
            self.use_cheb = True
            self.rf_backend = "hybrid"
        else:
            raise ValueError(
                f"Unknown rf_backend={rf_backend}, expected 'rff', 'chebyshev', or 'hybrid'."
            )

        # 1) Feature budget: D_k_base = 2 * rf_total_basis
        Dk_total_target = 2 * self.rf_total_basis   

        # RFF / Chebyshev will end up with basis/degree
        if self.rf_backend == "rff":
            # Original RFF-only: use num_basis as is
            self.num_basis_rff = self.num_basis_cfg
            self.taylor_degree_cheb = 0

        elif self.rf_backend == "chebyshev":
            # Chebyshev-only: use taylor_degree as configured
            self.num_basis_rff = 0
            self.taylor_degree_cheb = max(1, int(taylor_degree))

        else:  # hybrid: use budget
            frac = float(rf_rff_frac)
            # Avoid extreme values
            frac = max(0.05, min(0.95, frac))

            # Target feature count
            D_rff_target = int(round(Dk_total_target * frac))
            D_cheb_target = Dk_total_target - D_rff_target

            # RFF: D_rff ≈ 2 * num_basis_rff
            if D_rff_target < 2:
                D_rff_target = 2
            self.num_basis_rff = max(1, D_rff_target // 2)
            D_rff_actual = 2 * self.num_basis_rff

            # Chebyshev: D_cheb ≈ 2 * kernel_Q * taylor_degree
            if D_cheb_target <= 0:
                # Minimum degree 1 for extreme ratios
                self.taylor_degree_cheb = 1
            else:
                self.taylor_degree_cheb = max(1, D_cheb_target // (2 * kernel_Q))

            D_cheb_actual = 2 * kernel_Q * self.taylor_degree_cheb

        # Save degree for Chebyshev-only mode (0 for RFF-only)
        if self.rf_backend == "chebyshev":
            self.taylor_degree_cheb = max(1, int(taylor_degree))

        # 2) V projection, out projection, time shift
        self.v_proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.tact = nn.SiLU()
        self.tproj = nn.Linear(temb_dim, in_channels)
        self.tproj.weight.data = default_init()(self.tproj.weight.data.shape)
        nn.init.zeros_(self.tproj.bias)

        # 3) RFF backend
        if self.use_rff:
            self.kernel_rff = TimeDependentRFFKernel1D(
                kernel_type=kernel_type,
                Q=kernel_Q,
                temb_dim=temb_dim,
                t_hidden=kernel_time_hidden,
                enable_time_cond=enable_kernel_time_cond,
                num_heads=self.num_heads,
                num_basis=self.num_basis_rff,          # ★ 분배된 basis 사용
                bandwidth_init=bandwidth_init,
                hyp_len_init=hyp_len_init,
                rff_type=rff_type,
                enable_spatial_cond=enable_spatial_cond,
                x_hidden=spatial_hidden,
            )
            # RFF feature dim = 2 * num_basis_rff
            self.rff_feat_dim = 2 * self.num_basis_rff
        else:
            self.kernel_rff = None
            self.rff_feat_dim = 0

        # 4) Chebyshev/Taylor backend
        if self.use_cheb:
            # Chebyshev-only : use taylor_degree argument
            deg = self.taylor_degree_cheb if self.rf_backend == "hybrid" else taylor_degree
            self.kernel_cheb = TimeDependentChebyshevKernel1D(
                taylor_degree=deg,
                kernel_Q=kernel_Q,
                num_heads=num_heads,
                temb_dim=temb_dim,
                t_hidden=kernel_time_hidden,
                enable_time_cond=enable_kernel_time_cond,
                sigma_init_scale=bandwidth_init,
            )
            # Chebyshev feature dim (per head) = 2 * Q * D
            self.cheb_feat_dim = self.kernel_cheb.feature_dim
        else:
            self.kernel_cheb = None
            self.cheb_feat_dim = 0

        # 5) Time-Adaptive head-wise gating (Coupled)
        #    - only used in hybrid: g_rff(t,h) + g_cheb(t,h) = 1
        if self.use_rff and self.use_cheb:
            self.gate_proj = nn.Linear(temb_dim, self.num_heads)
            # 초기값 0 → sigmoid(0)=0.5 → RFF/Cheb 0.5/0.5로 시작 (기존 unified_gate_raw=0과 동일)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
        else:
            # single-backend mode: gate ≡ 1 (no additional computation)
            self.gate_proj = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1))
        else:
            self.bias = None

    # φ_RFF(x): (B, N) × temb → (B, N, H, Drff)
    def _phi_rff(self, x_coords: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        if self.kernel_rff is None:
            raise RuntimeError("RFF backend is disabled but _phi_rff was called.")

        B, N = x_coords.shape
        omega_x, amp_x = self.kernel_rff.make_omega_and_amp_spatial(
            x_coords, temb, device=x_coords.device, dtype=x_coords.dtype
        )  # (B, N, H, D_rff)

        phase = x_coords.view(B, N, 1, 1) * omega_x
        cos_part = torch.cos(phase)
        sin_part = torch.sin(phase)

        if amp_x is not None:
            cos_part = cos_part * amp_x
            sin_part = sin_part * amp_x

        phi = torch.cat([cos_part, sin_part], dim=-1)  # (B, N, H, 2*D_rff)
        # RFF normalization: 1/sqrt(num_basis_rff)  (maintain previous experimental compatibility)
        phi = phi * (float(self.num_basis_rff) ** -0.5)
        return phi

    # φ_Cheb(x): (B, N) × temb → (B, N, H, Dcheb)
    def _phi_cheb(self, x_coords: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        if self.kernel_cheb is None:
            raise RuntimeError("Chebyshev backend is disabled but _phi_cheb was called.")
        # TimeDependentChebyshevKernel1D.get_feature_map: (B, N, H, feature_dim)
        phi = self.kernel_cheb.get_feature_map(x_coords, temb)
        return phi

    # Unified RF feature map Φ(x)
    def _get_phi(self, x_coords: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Unified RF feature map
          - hybrid 모드: time-adaptive, head-wise coupled gate (g_rff + g_cheb = 1)
          - single-backend 모드: gate ≡ 1
        """
        B, N = x_coords.shape

        # 1) Create features for each RF family
        phi_rff, phi_cheb = None, None

        if self.use_rff:
            phi_rff = self._phi_rff(x_coords, temb)      # (B, N, H, D_rff)

        if self.use_cheb:
            phi_cheb = self._phi_cheb(x_coords, temb)    # (B, N, H, D_cheb)

        phis = []

        # 2) hybrid : time-dependent head-wise gate
        if self.gate_proj is not None:
            # temb: (B, C_temb) → (B, H)
            a = self.gate_proj(temb)                     # (B, H)
            g_rff = torch.sigmoid(a)                     # (B, H)
            g_cheb = 1.0 - g_rff                         # (B, H)

            # broadcast shape: (B, 1, H, 1)
            g_rff = g_rff.view(B, 1, self.num_heads, 1)
            g_cheb = g_cheb.view(B, 1, self.num_heads, 1)

            if phi_rff is not None:
                # sqrt(g) multiplies to kernel mixture weight
                phis.append(phi_rff * g_rff.sqrt())
            if phi_cheb is not None:
                phis.append(phi_cheb * g_cheb.sqrt())

        else:
            # single-backend: gate ≡ 1
            if phi_rff is not None:
                phis.append(phi_rff)
            if phi_cheb is not None:
                phis.append(phi_cheb)

        if not phis:
            raise RuntimeError("No RF backend is active in _get_phi().")

        if len(phis) == 1:
            return phis[0]
        return torch.cat(phis, dim=-1)  # (B, N, H, D_total)

    # forward: Linear Attention with Unified RF
    def forward(
        self,
        x: torch.Tensor,          # (B, C, N)
        x_coords: torch.Tensor,   # (B, N)
        temb: torch.Tensor,       # (B, C_temb)
    ) -> torch.Tensor:
        B, C, N = x.shape
        H = self.num_heads
        D_h = self.head_dim

        # 1) Time shift
        shift = self.tproj(self.tact(temb))        # (B, C)
        x = x + shift.unsqueeze(-1)                # (B, C, N)

        # 2) Value projection
        V = self.v_proj(x)                         # (B, C, N)
        V = V.view(B, H, D_h, N)                   # (B, H, D_h, N)

        # 3) Integration weights
        w = _trapz_weights_1d(x_coords) * self.measure_scale   # (B, N)
        V = V * w.view(B, 1, 1, N)                             # (B, H, D_h, N)
        V_lin = V.transpose(-2, -1).contiguous()                          # (B, H, N, D_h)

        # 4) Unified RF feature → Q, K
        phi = self._get_phi(x_coords, temb)                     # (B, N, H, D_k)
        phi = phi.permute(0, 2, 1, 3).contiguous()                           # (B, H, N, D_k)
        Q = phi
        K = phi

        # 5) Linear Attention
        BH = B * H
        D_k = Q.shape[-1]
        Q2 = Q.reshape(BH, N, D_k)                               # (BH, N, D_k)
        K2 = K.reshape(BH, N, D_k)                               # (BH, N, D_k)
        V2 = V_lin.reshape(BH, N, D_h)                           # (BH, N, D_h)
        # context = K^T @ V
        context = torch.bmm(K2.transpose(1, 2).contiguous(), V2) # (BH, D_k, D_h)
        # out = Q @ context
        out2 = torch.bmm(Q2, context)                            # (BH, N, D_h)
        out_heads = out2.view(B, H, N, D_h)                      # (B, H, N, D_h)

        # 6) heads concat + out projection
        out_heads = out_heads.permute(0, 1, 3, 2).contiguous().view(B, C, N)  # (B, C, N)
        out = self.out_proj(out_heads)
        if self.bias is not None:
            out = out + self.bias
        return out

class MHLKNO_LinAttn(nn.Module):
    """
    MHLKNO expressed as explicit Linear Attention(Q,K,V) form for 1D Hilbert Diffusion model.

    Similar to MHLKNO but replaces internal kernel layer with LinearAttentionKNO1D
    and separates Q, K, V at code level.
    """

    def __init__(self, config):
        super().__init__()
        cfg = config.model

        self.hidden_channels = cfg.hidden_channels
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.lifting_channels = cfg.lifting_channels
        self.projection_channels = cfg.projection_channels
        self.n_layers = cfg.n_layers

        # Multi-kernel heads / RFF basis
        self.num_heads = int(getattr(cfg, "num_kernel_heads", 4))
        self.num_basis = int(getattr(cfg, "num_kernel_basis", 100))
        self.bandwidth_init = float(getattr(cfg, "kernel_bandwidth_init", 1.0))
        self.rff_type = str(getattr(cfg, "kernel_rff_type", "ghq")).lower()

        #   - rf_backend: "rff" | "chebyshev" | "hybrid"
        #   - rf_total_basis: feature budget (default: num_kernel_basis)
        #   - rf_rff_frac  : ratio of RFF in hybrid
        self.rf_backend = str(getattr(cfg, "rf_backend", "rff")).lower()
        self.rf_total_basis = int(getattr(cfg, "rf_total_basis", self.num_basis))
        self.rf_rff_frac = float(getattr(cfg, "rf_rff_frac", 0.5))
        # Chebyshev-only mode: base taylor degree (recalculated in hybrid mode)
        self.taylor_degree = int(getattr(cfg, "taylor_degree", 12))      

        # Integral measure scale (Jacobian correction for coordinate normalization)
        measure_scale = float(getattr(cfg, "measure_scale", 1.0))

        self.norm_type = getattr(cfg, "norm", "group_norm")
        self.preactivation = getattr(cfg, "preactivation", True)
        self.skip_type = getattr(cfg, "skip", "soft-gating")

        # Kernel hyperparameters
        self.kernel_type = str(getattr(cfg, "kernel_type", "gsm")).lower()
        self.kernel_Q = int(getattr(cfg, "kernel_Q", 6))
        self.kernel_hidden = int(getattr(cfg, "kernel_hidden", 64))
        self.enable_kernel_time_cond = bool(getattr(cfg, "enable_kernel_time_cond", True))
        self.kernel_time_hidden = int(getattr(cfg, "kernel_time_hidden", 64))
        hyp_len_init = getattr(config.data, "hyp_len", 0.2)
        self.enable_spatial_kernel = bool(getattr(cfg, "enable_spatial_kernel", False))
        self.spatial_kernel_hidden = int(getattr(cfg, "spatial_kernel_hidden", 64))

        if self.hidden_channels % self.num_heads != 0:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must be divisible by num_kernel_heads ({self.num_heads})"
            )

        # Time embedding → channel shift
        self.Dense = nn.ModuleList([
            nn.Linear(self.lifting_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels,   self.hidden_channels),
        ])
        for layer in self.Dense:
            layer.weight.data = default_init()(layer.weight.data.shape)
            nn.init.zeros_(layer.bias)

        # Lifting / Projection: same as FNO skeleton
        self.lifting = Lifting(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            n_dim=1,
        )
        self.projection = Projection(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_dim=1,
            non_linearity=F.gelu,
        )

        # KNO layer composed of LinearAttentionKNO1D
        self.layers = nn.ModuleList([
            LinearAttentionKNO1D(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                num_heads=self.num_heads,
                num_basis=self.num_basis,             
                temb_dim=self.lifting_channels,
                bandwidth_init=self.bandwidth_init,
                measure_scale=measure_scale,
                # Unified RF backend 
                rf_backend=self.rf_backend,
                rf_total_basis=self.rf_total_basis,
                rf_rff_frac=self.rf_rff_frac,
                # 공통 kernel hyper
                kernel_type=self.kernel_type,
                kernel_Q=self.kernel_Q,
                kernel_hidden=self.kernel_hidden,
                enable_kernel_time_cond=self.enable_kernel_time_cond,
                kernel_time_hidden=self.kernel_time_hidden,
                hyp_len_init=hyp_len_init,
                rff_type=self.rff_type,
                bias=True,
                enable_spatial_cond=self.enable_spatial_kernel,
                spatial_hidden=self.spatial_kernel_hidden,
                # Chebyshev-only 모드용 base taylor_degree
                taylor_degree=self.taylor_degree,
            )
            for _ in range(self.n_layers)
        ])

        # Norm / Skip / MLP
        if self.norm_type == "group_norm":
            self.norms = nn.ModuleList([
                nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels)
                for _ in range(self.n_layers)
            ])
        else:
            self.norms = None

        self.skips = nn.ModuleList([
            skip_connection(
                self.hidden_channels,
                self.hidden_channels,
                n_dim=1,
                type=self.skip_type,
            )
            for _ in range(self.n_layers)
        ])

        self.mlps = nn.ModuleList([
            MLP(
                in_channels=self.hidden_channels,
                hidden_channels=int(round(self.hidden_channels * 4.0)),
                dropout=0.0,
                n_dim=1,
                temb_dim=self.hidden_channels,
            )
            for _ in range(self.n_layers)
        ])
    def all_kappas(self):
        return []

    def all_baselines(self):
        return []

    def all_band_gates(self):
        return []

    # ------------------------------------------------------------------
    # forward: same as Hilbert loss interface
    #   x : (B, 2, N)  = [signal, coord_norm]
    #   t : (B,)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Coordinate channel (normalized [-1,1]) is preserved separately
        x_coord_norm = x[:, -1, :]                # (B, N)

        # Lifting: (B, 2, N) -> (B, C_hid, N)
        h = self.lifting(x)

        # Time embedding
        temb = get_timestep_embedding(t, self.lifting_channels)   # (B, lifting_channels)
        temb = self.Dense[0](temb)
        temb = self.Dense[1](F.silu(temb))
        h = h + temb.unsqueeze(-1)             # (B, C_hid, N)

        # Layer stack
        for i in range(self.n_layers):
            if self.preactivation:
                h = F.silu(h)
                if self.norms is not None:
                    h = self.norms[i](h)

            # Linear Attention-based KNO layer
            h_k = self.layers[i](h, x_coord_norm, temb)

            if not self.preactivation and self.norms is not None:
                h_k = self.norms[i](h_k)

            # Residual + MLP
            h = h_k + self.skips[i](h)
            h = self.mlps[i](h, temb)

        y = self.projection(h).squeeze(1)       # (B, N)
        return y
