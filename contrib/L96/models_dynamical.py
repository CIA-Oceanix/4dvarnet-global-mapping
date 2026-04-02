import torch
import torch.nn as nn
import torch.nn.functional as F

class Lorenz96ODEPrior(nn.Module):
    """
    Lorenz-96 dynamical prior for 4DVarNet.

    Drop-in replacement for BilinAEPriorCost under GradSolver.prior_cost.

    Norm stats are injected after construction via set_norm_stats(), called
    from Lit4dVarNet.setup() once the datamodule is attached. Until then,
    buffers default to (0, 1) so the module is safe to instantiate from Hydra
    without a trainer present.

    State convention (matches GradSolver):
        normalized:  x̃ = (x - μ) / σ
        physical:    x = x̃ · σ + μ
    Input shape: [B, 1, T, D]
    """

    def __init__(
        self,
        dt: float = 0.05,
        F_param: float = 8.0,
        scheme: str = "rk4",
    ):
        super().__init__()
        self.dt = dt
        self.scheme = scheme

        # F is a non-optimized parameter so it travels with state_dict
        self.F = nn.Parameter(torch.tensor(float(F_param)), requires_grad=False)

        # Safe defaults — overwritten by set_norm_stats() before training
        self.register_buffer("norm_mean", torch.tensor(0.0))
        self.register_buffer("norm_std",  torch.tensor(1.0))
        self._norm_stats_set = False  # sentinel for debugging

    # ------------------------------------------------------------------
    # Norm-stats injection (called from Lit4dVarNet.setup)
    # ------------------------------------------------------------------

    def set_norm_stats(self, mean: float, std: float) -> None:
        device = self.norm_mean.device
        self.norm_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.norm_std  = torch.tensor(std,  dtype=torch.float32, device=device)
        self._norm_stats_set = True

    def _warn_if_stats_missing(self) -> None:
        if not self._norm_stats_set:
            import warnings
            warnings.warn(
                "Lorenz96ODEPrior: norm_stats not set — using (0, 1). "
                "Call set_norm_stats() or ensure Lit4dVarNet.setup() runs first.",
                UserWarning, stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Norm helpers (broadcast-safe for [B, 1, T, D])
    # ------------------------------------------------------------------

    def _denorm(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.norm_std + self.norm_mean

    def _renorm(self, x_phys: torch.Tensor) -> torch.Tensor:
        return (x_phys - self.norm_mean) / self.norm_std

    # ------------------------------------------------------------------
    # Lorenz-96 RHS — physical space, periodic BC via torch.roll
    # ------------------------------------------------------------------

    def _rhs(self, x: torch.Tensor) -> torch.Tensor:
        xm2 = torch.roll(x,  2, dims=-1)
        xm1 = torch.roll(x,  1, dims=-1)
        xp1 = torch.roll(x, -1, dims=-1)
        return (xp1 - xm2) * xm1 - x + self.F

    def _euler_step(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dt * self._rhs(x)

    def _rk4_step(self, x: torch.Tensor) -> torch.Tensor:
        k1 = self._rhs(x)
        k2 = self._rhs(x + 0.5 * self.dt * k1)
        k3 = self._rhs(x + 0.5 * self.dt * k2)
        k4 = self._rhs(x + self.dt * k3)
        return x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        if self.scheme == "euler":
            return self._euler_step(x)
        if self.scheme == "rk4":
            return self._rk4_step(x)
        raise ValueError(f"Unknown scheme: {self.scheme!r}")

    # ------------------------------------------------------------------
    # ODE rollout in physical space
    # ------------------------------------------------------------------

    def _rollout_physical(self, x_phys: torch.Tensor) -> torch.Tensor:
        """[B,1,T,D] physical → [B,1,T,D] predicted physical trajectory."""
        x_t    = x_phys[:, :, :-1, :]        # all but last timestep
        x_tp1  = self._step(x_t)             # one-step prediction
        return torch.cat([x_phys[:, :, :1, :], x_tp1], dim=2)

    # ------------------------------------------------------------------
    # Public API — mirrors BilinAEPriorCost interface
    # ------------------------------------------------------------------

    def forward_ae(self, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Normalized → ODE-projected normalized state.
        Called by GradSolver.forward() at inference time
        (replaces prior_cost.forward_ae in the original GradSolver).
        """
        return self._renorm(self._rollout_physical(self._denorm(state_norm)))

    def forward(self, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Prior cost scalar.
        Called by Lit4dVarNet.step() as: self.solver.prior_cost(state)
        Loss is MSE in normalized space for consistent gradient scale.
        """
        self._warn_if_stats_missing()
        pred_norm = self.forward_ae(state_norm)
        return F.mse_loss(state_norm, pred_norm)
