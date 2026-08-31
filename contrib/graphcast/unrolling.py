"""GraphCast update modules for pure-gradient variational unrolling.

This module intentionally composes :class:`GraphCastSLASolver` without changing
the direct-inversion implementation.  It provides the small interface adapters
needed by the variational solver and keeps all graph-specific state (coordinates
and step conditioning) local to this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class FourierStepEmbedding(nn.Module):
    """Embed a normalized unrolling step with dyadic Fourier features."""

    def __init__(self, num_frequencies: int = 8, output_dim: int = 16):
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.num_frequencies = int(num_frequencies)
        self.output_dim = int(output_dim)
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32),
            persistent=False,
        )
        feature_dim = 2 * self.num_frequencies
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    @staticmethod
    def _step_vector(timesteps: torch.Tensor | float) -> torch.Tensor:
        timesteps = torch.as_tensor(timesteps)
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        elif timesteps.ndim == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        elif timesteps.ndim != 1:
            raise ValueError(
                "timesteps must be a scalar or have shape (B,) or (B, 1), "
                f"got {tuple(timesteps.shape)}"
            )
        return timesteps

    def fourier_features(self, timesteps: torch.Tensor | float) -> torch.Tensor:
        """Return ``sin``/``cos`` features at frequencies ``2**0 .. 2**7``."""
        timesteps = self._step_vector(timesteps).to(
            device=self.frequencies.device,
            dtype=self.frequencies.dtype,
        )
        angles = 2.0 * torch.pi * timesteps[:, None] * self.frequencies[None, :]
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def forward(self, timesteps: torch.Tensor | float) -> torch.Tensor:
        return self.projection(self.fourier_features(timesteps))


class TimeConditionedGraphCastUpdate(nn.Module):
    """Adapt a direct GraphCast model into a time-conditioned update operator."""

    def __init__(
        self,
        graph_model: nn.Module,
        num_frequencies: int = 8,
        time_feature_channels: int = 16,
        checkpoint_update: bool = True,
        zero_initialize: bool = True,
    ):
        super().__init__()
        if num_frequencies != 8:
            raise ValueError("GraphCast unrolling requires exactly 8 frequencies")
        if time_feature_channels != 16:
            raise ValueError("GraphCast unrolling requires 16 time-feature channels")
        if not isinstance(graph_model, nn.Module):
            raise TypeError("graph_model must be a torch.nn.Module")

        self.graph_model = graph_model
        self.checkpoint_update = bool(checkpoint_update)
        self.time_feature_channels = int(time_feature_channels)
        self.time_embedding = FourierStepEmbedding(
            num_frequencies=num_frequencies,
            output_dim=time_feature_channels,
        )

        self._validate_graph_model()
        if zero_initialize:
            self._zero_initialize_decoder()

    @property
    def time_projection(self) -> nn.Sequential:
        """Expose the learned time projection for inspection and optimization."""
        return self.time_embedding.projection

    def _validate_graph_model(self) -> None:
        required = (
            "in_channels",
            "out_channels",
            "use_input_mask",
            "predict_residual",
            "decoder",
        )
        missing = [name for name in required if not hasattr(self.graph_model, name)]
        if missing:
            raise TypeError(
                "graph_model is missing required GraphCast attributes: "
                + ", ".join(missing)
            )
        if self.graph_model.in_channels != 31:
            raise ValueError(
                "unrolling GraphCast must have in_channels=31 "
                "(15 gradients + 16 step features)"
            )
        if self.graph_model.out_channels != 15:
            raise ValueError("unrolling GraphCast must have out_channels=15")
        if self.graph_model.use_input_mask:
            raise ValueError("unrolling GraphCast requires use_input_mask=False")
        if self.graph_model.predict_residual:
            raise ValueError("unrolling GraphCast requires predict_residual=False")

    def _zero_initialize_decoder(self) -> None:
        linear_layers = [
            module
            for module in self.graph_model.decoder.modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            raise TypeError("graph_model.decoder must contain a final nn.Linear layer")
        final_layer = linear_layers[-1]
        nn.init.zeros_(final_layer.weight)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)

    def fourier_features(self, timesteps: torch.Tensor | float) -> torch.Tensor:
        return self.time_embedding.fourier_features(timesteps)

    @staticmethod
    def _normalize_timesteps(
        timesteps: torch.Tensor | float | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if timesteps is None:
            raise ValueError("timesteps are required for GraphCast unrolling")
        timesteps = torch.as_tensor(timesteps, device=device, dtype=torch.float32)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.ndim == 2 and timesteps.shape == (batch_size, 1):
            timesteps = timesteps[:, 0]
        elif timesteps.ndim == 1 and timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.ndim != 1 or timesteps.numel() != batch_size:
            raise ValueError(
                f"expected one timestep per sample ({batch_size}), "
                f"got shape {tuple(timesteps.shape)}"
            )
        if not torch.isfinite(timesteps).all():
            raise ValueError("timesteps contain non-finite values")
        if not ((timesteps >= 0.0) & (timesteps <= 1.0)).all():
            raise ValueError("normalized timesteps must lie in [0, 1]")
        return timesteps

    @staticmethod
    def _validate_coordinate(
        values: Any,
        name: str,
        expected_size: int,
        batch_size: int,
    ) -> torch.Tensor:
        try:
            coordinates = torch.as_tensor(values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"extra['{name}'] must be array-like coordinates") from exc

        if coordinates.ndim == 2:
            if coordinates.shape != (batch_size, expected_size):
                raise ValueError(
                    f"extra['{name}'] has incompatible shape "
                    f"{tuple(coordinates.shape)}; expected "
                    f"({batch_size}, {expected_size})"
                )
            reference = coordinates[0]
            if not torch.equal(
                coordinates,
                reference.unsqueeze(0).expand_as(coordinates),
            ):
                raise ValueError(
                    f"all samples in a batch must share the same {name} coordinates"
                )
        elif coordinates.ndim != 1 or coordinates.numel() != expected_size:
            raise ValueError(
                f"expected extra['{name}'] with {expected_size} values, "
                f"got shape {tuple(coordinates.shape)}"
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError(f"extra['{name}'] contains non-finite coordinates")
        return coordinates

    def _conditioned_update(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
    ) -> torch.Tensor:
        step_features = self.time_embedding(timesteps).to(dtype=x.dtype)
        step_features = step_features[:, :, None, None].expand(
            -1,
            -1,
            x.shape[-2],
            x.shape[-1],
        )
        graph_input = torch.cat((x, step_features), dim=1)
        graph_batch = SimpleNamespace(
            input=graph_input,
            lat=latitudes,
            lon=longitudes,
        )
        return self.graph_model(graph_batch)

    def predict(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        return self.forward(x, timesteps=timesteps, extra=extra)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"expected gradient with shape (B, 15, H, W), got {tuple(x.shape)}"
            )
        if x.shape[1] != 15:
            raise ValueError(f"expected 15 gradient channels, got {x.shape[1]}")
        if not isinstance(extra, Mapping):
            raise ValueError("extra must be a mapping containing 'lat' and 'lon'")
        missing = [name for name in ("lat", "lon") if name not in extra]
        if missing:
            raise ValueError(
                "GraphCast unrolling requires coordinates in extra: missing "
                + ", ".join(missing)
            )

        batch_size, _, height, width = x.shape
        normalized_steps = self._normalize_timesteps(
            timesteps,
            batch_size=batch_size,
            device=x.device,
        )
        latitudes = self._validate_coordinate(
            extra["lat"],
            "lat",
            expected_size=height,
            batch_size=batch_size,
        )
        longitudes = self._validate_coordinate(
            extra["lon"],
            "lon",
            expected_size=width,
            batch_size=batch_size,
        )

        if self.training and self.checkpoint_update:
            return checkpoint(
                self._conditioned_update,
                x,
                normalized_steps,
                latitudes,
                longitudes,
                use_reentrant=False,
            )
        return self._conditioned_update(
            x,
            normalized_steps,
            latitudes,
            longitudes,
        )


class GraphGlobalGradModelWithCondition(nn.Module):
    """Normalize the first total-cost gradient and apply the graph update model."""

    def __init__(
        self,
        grad_model: nn.Module,
        dropout: float = 0.0,
        use_grad_norm: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not isinstance(grad_model, nn.Module):
            raise TypeError("grad_model must be a torch.nn.Module")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.grad_model = grad_model
        self.dropout = nn.Dropout(dropout)
        self.use_grad_norm = bool(use_grad_norm)
        self.eps = float(eps)
        self._grad_norm: torch.Tensor | None = None

    def reset_state(self, inp: torch.Tensor | None = None) -> None:
        del inp
        self._grad_norm = None

    def _first_gradient_rms(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate low-precision gradients in fp32.  The cast and clamp remain
        # in the autograd graph; the normalization is deliberately not detached.
        rms_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean_square = rms_input.square().mean()
        return mean_square.clamp_min(self.eps**2).sqrt()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        if self._grad_norm is None:
            if self.use_grad_norm:
                self._grad_norm = self._first_gradient_rms(x)
            else:
                self._grad_norm = x.new_ones(())
        normalized = x / self._grad_norm
        normalized = self.dropout(normalized).to(dtype=x.dtype)
        return self.grad_model.predict(
            normalized,
            timesteps=timesteps,
            extra=extra,
        )


class GraphGlobalUnrollingSolver(nn.Module):
    """Five-step-compatible pure total-gradient variational solver."""

    def __init__(
        self,
        prior_cost: nn.Module,
        obs_cost: nn.Module,
        grad_mod: nn.Module,
        n_step: int,
        std_init: float = 0.1,
        lbd: float = 1.0,
        lr_grad: float = 0.0,
        input_grad_update: str = "grad",
    ):
        super().__init__()
        if not isinstance(n_step, int) or isinstance(n_step, bool) or n_step <= 0:
            raise ValueError("n_step must be a positive integer")
        if std_init < 0.0:
            raise ValueError("std_init must be non-negative")
        if input_grad_update != "grad":
            raise ValueError(
                "GraphGlobalUnrollingSolver currently supports only "
                "input_grad_update='grad'"
            )
        if float(lr_grad) != 0.0:
            raise ValueError(
                "GraphGlobalUnrollingSolver requires lr_grad=0; "
                "the learned GraphCast update is the complete step"
            )

        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod
        self.n_step = n_step
        self.std_init = float(std_init)
        self.lbd = float(lbd)
        self.lr_grad = float(lr_grad)
        self.input_grad_update = input_grad_update

    def init_state(self, batch: Any, x_init: torch.Tensor | None = None) -> torch.Tensor:
        if not hasattr(batch, "input"):
            raise ValueError("batch must provide an input tensor")
        if x_init is not None:
            if x_init.shape != batch.input.shape:
                raise ValueError(
                    f"x_init shape {tuple(x_init.shape)} does not match "
                    f"batch.input shape {tuple(batch.input.shape)}"
                )
            return x_init.detach().requires_grad_(True)
        state = self.std_init * torch.randn_like(batch.input)
        return state.detach().requires_grad_(True)

    @staticmethod
    def _coordinates(batch: Any) -> dict[str, Any]:
        missing = [name for name in ("lat", "lon") if not hasattr(batch, name)]
        if missing:
            raise ValueError(
                "GraphCast unrolling requires batch coordinates: missing "
                + ", ".join(missing)
            )
        return {"lat": batch.lat, "lon": batch.lon}

    def forward(
        self,
        batch: Any,
        x_init: torch.Tensor | None = None,
        phase: str | None = None,
    ) -> torch.Tensor:
        del phase
        with torch.enable_grad():
            state = self.init_state(batch, x_init=x_init)
            extra = self._coordinates(batch)
            self.grad_mod.reset_state(batch.input)

            for step_index in range(self.n_step):
                variational_cost = self.prior_cost(state)
                variational_cost = variational_cost + self.lbd**2 * self.obs_cost(
                    state,
                    batch,
                )
                gradient = torch.autograd.grad(
                    variational_cost,
                    state,
                    create_graph=self.training,
                )[0]
                timesteps = state.new_full(
                    (state.shape[0],),
                    float(step_index) / self.n_step,
                )
                update = self.grad_mod(
                    gradient,
                    timesteps=timesteps,
                    extra=extra,
                )
                if update.shape != state.shape:
                    raise ValueError(
                        f"gradient model returned shape {tuple(update.shape)}; "
                        f"expected {tuple(state.shape)}"
                    )
                state = state - update / self.n_step

                if not self.training:
                    state = state.detach().requires_grad_(True)

        return state


__all__ = [
    "FourierStepEmbedding",
    "TimeConditionedGraphCastUpdate",
    "GraphGlobalGradModelWithCondition",
    "GraphGlobalUnrollingSolver",
]
