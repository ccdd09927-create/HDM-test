import torch
import math

class QuadraticDataset(torch.utils.data.Dataset):
    """
    func_type ∈ {
        'quadratic', 'linear', 'circle', 'sin', 'doppler', 'sinc',
        'gaussian_bumps', 'am_sin'
    } 
      - quadratic: y = a * x^2 + ε,  a∈{-1,+1}
      - linear   : y = a * x   + ε,  a∈{-1,+1}
      - circle   : y = a * sqrt(r^2 - x^2) + ε,  a∈{-1,+1},  r∈{10,5} (domain [-10,10])
      - sin      : y = sin(x) + ε
      - sinc     : y = sinc(x) + ε, sinc(x)=sin(πx)/(πx)  (domain [-10,10])
      - doppler  : y = sqrt(x(1-x)) * sin((2π*1.05)/(x+0.05)) + b
                   (domain [0,1], b~N(0, noise_std^2) is "function-specific offset")
      - gaussian_bumps:
            y = Σ_{i=1}^K A_i exp(-0.5 * ((x-c_i)/σ_i)^2) + ε
            (K=24 fixed, {c_i,A_i,σ_i} are fixed single-sample parameters by seed)
      - am_sin (amplitude-modulated sinusoid):
            envelope(x)= (1 + d_1 cos(ω_{m1} x + 0.3)) (1 + d_2 cos(ω_{m2} x - 1.1))
            taper(x)=0.55 + 0.45 exp(-0.5 (x/8)^2)
            y = taper(x) * envelope(x) * sin(ω_c x + φ) + ε
            (ω_c=2.3 fixed, ω_{m1}=0.55, ω_{m2}=0.18, d_1=0.95, d_2=0.65, φ is fixed single-sample phase by seed)

    Saved form  : Z-normalized values for entire dataset (mean/std are calculated over the entire dataset)
    Inverse transformation: inverse_transform(y_norm) = y_norm * std + mean

    Coordinate normalization (matches runner):
      - {quadratic, linear, sin, circle, sinc, gaussian_bumps, am_sin}:  x_norm = x / 10 ∈ [-1,1]
      - doppler:  x_norm = (x - 0.5) / 0.5 ∈ [-1,1]
    """
    def __init__(self, num_data: int, num_points: int, seed: int = 42,
                 grid_type: str = 'uniform', noise_std: float = 0.0,
                 func_type: str = 'quadratic'):
        super().__init__()
        torch.manual_seed(seed)

        func_type = str(func_type).lower()
        if func_type not in (
            'quadratic', 'linear', 'circle', 'sin', 'doppler', 'sinc',
            'gaussian_bumps', 'am_sin'
        ):
            raise ValueError(
                "func_type must be one of "
                "'quadratic','linear','circle','sin','doppler','sinc','gaussian_bumps','am_sin', "
                f"got {func_type}"
            )
        self.func_type  = func_type

        self.num_data   = num_data
        self.num_points = num_points
        self.seed       = seed
        self.grid_type  = grid_type
        self.is_train   = True
        self.noise_std  = float(noise_std)

        # Coordinate/domain setting (matches runner's _coord_norm)
        if self.func_type in ('doppler',):
            # [0,1] → [-1,1]
            self.coord_scale  = 0.5
            self.coord_offset = 0.5
            self._xmin, self._xmax = 0.0, 1.0
        else:
            # [-10,10] → [-1,1]
            self.coord_scale  = 10.0
            self.coord_offset = 0.0
            self._xmin, self._xmax = -10.0, 10.0

        # Circle radius (matches domain [-10,10])
        self.radius = 10.0
        self.circle_radii = torch.tensor([10.0, 5.0])

        # Linear slope candidates: a ∈ {-1, 0, 1}
        self.linear_coeffs = torch.tensor([-1.0, 0.0, 1.0])

        # ─────────────────────────────────────────────────────────
        # (0) Fixed single-sample parameters (gaussian_bumps / am_sin)
        #     Use separate Generator to minimize global RNG interference
        # ─────────────────────────────────────────────────────────
        if self.func_type == 'gaussian_bumps':
            g = torch.Generator(device='cpu').manual_seed(self.seed + 1701)
            self.gb_num_bumps = 24

            self.gb_sigma_scale = 3.0

            span = (self._xmax - 0.7) - (self._xmin + 0.7)
            self.gb_centers = (self._xmin + 0.7) + torch.rand(self.gb_num_bumps, generator=g) * span
            self.gb_amps = torch.randn(self.gb_num_bumps, generator=g)

            # sigmas (pre-scale): log-uniform in [0.10, 0.60]
            sigma_min, sigma_max = 0.10, 0.60
            u = torch.rand(self.gb_num_bumps, generator=g)
            log_min = math.log10(sigma_min)
            log_max = math.log10(sigma_max)
            self.gb_sigmas = torch.pow(10.0, log_min + (log_max - log_min) * u) * self.gb_sigma_scale

        if self.func_type == 'am_sin':
            g = torch.Generator(device='cpu').manual_seed(self.seed + 2909)
            self.am_carrier_w = 2.3
            self.am_mod_w = 0.55      # ω_{m1}
            self.am_mod_w2 = 0.18     # ω_{m2} 
            self.am_mod_depth = 0.95  # d_1
            self.am_mod_depth2 = 0.65 # d_2
            self.am_phase = float(2.0 * math.pi * torch.rand((), generator=g))

        # 1) Coordinate grid generation
        if grid_type == 'uniform':
            x_base = torch.linspace(start=self._xmin, end=self._xmax, steps=self.num_points)
        elif grid_type == 'random':
            x_base = (torch.rand(self.num_points) * (self._xmax - self._xmin) + self._xmin).sort().values
        else:
            raise ValueError(f"Unknown grid_type: '{grid_type}'. Choose 'uniform' or 'random'")

        self.x = x_base.unsqueeze(0).repeat(self.num_data, 1)  # (B, N)

        # 2) Data generation (ε or b: function-specific constant noise)
        torch.manual_seed(self.seed)

        if self.func_type == 'doppler':
            # b: (B,1) → (B,N)
            b = torch.randn(self.num_data, 1) * self.noise_std
            b = b.repeat(1, self.num_points)

            xx = self.x.clamp(0.0, 1.0)
            amp   = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
            y = amp * torch.sin(phase) + b

        else:
            # Common: function-specific constant noise ε
            eps = torch.randn(self.num_data, 1) * self.noise_std
            eps = eps.repeat(1, self.num_points)

            if self.func_type == 'quadratic':
                a = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)
                y = a * (self.x ** 2) + eps

            elif self.func_type == 'linear':
                a_choices = self.linear_coeffs.to(self.x.device).type_as(self.x)
                a_idx = torch.randint(low=0, high=a_choices.numel(), size=(self.num_data, 1), device=self.x.device)
                a = a_choices[a_idx].repeat(1, self.num_points)
                y = a * self.x + eps

            elif self.func_type == 'sin':
                y = torch.sin(self.x) + eps

            elif self.func_type == 'sinc':
                y = torch.sinc(self.x) + eps

            elif self.func_type == 'gaussian_bumps':
                # Create single-sample base function in (1,N) and repeat to (B,N)
                x0 = self.x[0:1, :]  # (1,N)
                centers = self.gb_centers.view(1, 1, -1)  # (1,1,K)
                amps    = self.gb_amps.view(1, 1, -1)     # (1,1,K)
                sigmas  = self.gb_sigmas.view(1, 1, -1)   # (1,1,K)

                diff = (x0.unsqueeze(-1) - centers) / sigmas     # (1,N,K)
                y0   = (amps * torch.exp(-0.5 * diff.pow(2))).sum(dim=-1)  # (1,N)
                y    = y0.repeat(self.num_data, 1) + eps

            elif self.func_type == 'am_sin':
                x0 = self.x[0:1, :]  # (1,N)
                envelope = (
                    1.0 + self.am_mod_depth * torch.cos(self.am_mod_w * x0 + 0.3)
                ) * (
                    1.0 + self.am_mod_depth2 * torch.cos(self.am_mod_w2 * x0 - 1.1)
                )
                y0 = envelope * torch.sin(self.am_carrier_w * x0 + self.am_phase)
                taper = torch.exp(-0.5 * (x0 / 8.0).pow(2)) * 0.45 + 0.55
                y0 = y0 * taper
                y = y0.repeat(self.num_data, 1) + eps

            else:  # 'circle'
                a = (torch.randint(low=0, high=2, size=(self.num_data, 1)) * 2 - 1).repeat(1, self.num_points)
                r_choices = self.circle_radii.to(self.x.device)
                r_idx = torch.randint(low=0, high=r_choices.numel(), size=(self.num_data, 1), device=self.x.device)
                r = r_choices[r_idx]  # (B,1)
                y = a * torch.sqrt((r.pow(2) - self.x.pow(2)).clamp_min(0.0)) + eps

        # 3) Z-normalization statistics and save
        self.mean = y.mean()
        self.std  = y.std().clamp_min(1e-8)
        self.dataset = (y - self.mean) / self.std

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        """
        grid_type=='random' & train: New coordinate/function sample for each call (resolution-independent learning)
        """
        if self.grid_type == 'random' and getattr(self, 'is_train', False):
            x_item = (torch.rand(self.num_points) * (self._xmax - self._xmin) + self._xmin).sort().values

            if self.func_type == 'doppler':
                b = (torch.randn(1) * self.noise_std).repeat(self.num_points)
                xx = x_item.clamp(0.0, 1.0)
                amp   = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
                phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
                y_item = amp * torch.sin(phase) + b

            else:
                eps = torch.randn(1).repeat(self.num_points) * self.noise_std

                if self.func_type == 'quadratic':
                    a = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
                    y_item = a * (x_item ** 2) + eps

                elif self.func_type == 'linear':
                    a_choices = self.linear_coeffs.to(x_item.device).type_as(x_item)
                    a = a_choices[torch.randint(low=0, high=a_choices.numel(), size=(1,), device=x_item.device)].item()
                    y_item = a * x_item + eps

                elif self.func_type == 'sin':
                    y_item = torch.sin(x_item) + eps

                elif self.func_type == 'sinc':
                    y_item = torch.sinc(x_item) + eps

                elif self.func_type == 'gaussian_bumps':
                    dev = x_item.device
                    centers = self.gb_centers.to(dev)  # (K,)
                    amps    = self.gb_amps.to(dev)     # (K,)
                    sigmas  = self.gb_sigmas.to(dev)   # (K,)

                    diff = (x_item.unsqueeze(-1) - centers.view(1, -1)) / sigmas.view(1, -1)  # (N,K)
                    y0 = (amps.view(1, -1) * torch.exp(-0.5 * diff.pow(2))).sum(dim=-1)       # (N,)
                    y_item = y0 + eps

                elif self.func_type == 'am_sin':
                    envelope = (
                        1.0 + self.am_mod_depth * torch.cos(self.am_mod_w * x_item + 0.3)
                    ) * (
                        1.0 + self.am_mod_depth2 * torch.cos(self.am_mod_w2 * x_item - 1.1)
                    )
                    y0 = envelope * torch.sin(self.am_carrier_w * x_item + self.am_phase)
                    taper = torch.exp(-0.5 * (x_item / 8.0).pow(2)) * 0.45 + 0.55
                    y_item = (y0 * taper) + eps

                else:  # 'circle'
                    a = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).item()
                    r_choices = self.circle_radii.to(x_item.device)
                    r = r_choices[torch.randint(low=0, high=r_choices.numel(), size=(1,))].item()
                    r2 = r * r
                    y_item = a * torch.sqrt((r2 - x_item.pow(2)).clamp_min(0.0)) + eps

            y_item = (y_item - self.mean) / self.std
            return x_item.unsqueeze(-1), y_item.unsqueeze(-1)

        # Fixed grid sample
        return (
            self.x[idx, :].unsqueeze(-1),
            self.dataset[idx, :].unsqueeze(-1)
        )

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)

    # ─────────────────────────────────────────────────────────
    # Generate original-scale function for resolution-independent validation/sampling
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_raw(self, x_batch: torch.Tensor, *, device=None) -> torch.Tensor:
        """
        x_batch: (B, N) — original coordinate values
          - doppler: x ∈ [0,1]
          - others : x ∈ [-10,10]
        return:  (B, N) — y_raw (before normalization)
        """
        dev = device or x_batch.device
        x_batch = x_batch.to(dev)
        B, N = x_batch.shape

        if self.func_type == 'doppler':
            b = torch.randn(B, 1, device=dev) * self.noise_std
            b = b.repeat(1, N)
            xx = torch.clamp(x_batch, 0.0, 1.0)
            amp   = torch.sqrt((xx * (1.0 - xx)).clamp_min(0.0))
            phase = (2.0 * math.pi * 1.05) / (xx + 0.05)
            y_raw = amp * torch.sin(phase) + b
            return y_raw

        # 공통(비 Doppler)
        eps = torch.randn(B, 1, device=dev) * self.noise_std
        eps = eps.repeat(1, N)

        if self.func_type == 'quadratic':
            a   = (torch.randint(low=0, high=2, size=(B, 1), device=dev) * 2 - 1).repeat(1, N)
            y_raw = a * (x_batch ** 2) + eps
            return y_raw

        if self.func_type == 'linear':
            a_choices = self.linear_coeffs.to(dev).type_as(x_batch)
            a_idx = torch.randint(low=0, high=a_choices.numel(), size=(B, 1), device=dev)
            a = a_choices[a_idx].repeat(1, N)
            y_raw = a * x_batch + eps
            return y_raw

        if self.func_type == 'sin':
            return torch.sin(x_batch) + eps

        if self.func_type == 'sinc':
            return torch.sinc(x_batch) + eps

        if self.func_type == 'gaussian_bumps':
            centers = self.gb_centers.to(dev).view(1, 1, -1)  # (1,1,K)
            amps    = self.gb_amps.to(dev).view(1, 1, -1)
            sigmas  = self.gb_sigmas.to(dev).view(1, 1, -1)

            diff = (x_batch.unsqueeze(-1) - centers) / sigmas         # (B,N,K)
            y0   = (amps * torch.exp(-0.5 * diff.pow(2))).sum(dim=-1) # (B,N)
            return y0 + eps

        if self.func_type == 'am_sin':
            envelope = (
                1.0 + self.am_mod_depth * torch.cos(self.am_mod_w * x_batch + 0.3)
            ) * (
                1.0 + self.am_mod_depth2 * torch.cos(self.am_mod_w2 * x_batch - 1.1)
            )
            y0 = envelope * torch.sin(self.am_carrier_w * x_batch + self.am_phase)
            taper = torch.exp(-0.5 * (x_batch / 8.0).pow(2)) * 0.45 + 0.55
            y0 = y0 * taper
            return y0 + eps

        # circle
        a = (torch.randint(low=0, high=2, size=(B, 1), device=dev) * 2 - 1).repeat(1, N)
        r_choices = self.circle_radii.to(dev)
        r_idx = torch.randint(low=0, high=r_choices.numel(), size=(B, 1), device=dev)
        r = r_choices[r_idx]  # (B,1)
        y_raw = a * torch.sqrt((r.pow(2) - x_batch.pow(2)).clamp_min(0.0)) + eps
        return y_raw
















