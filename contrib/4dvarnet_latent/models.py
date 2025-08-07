import functools as ft
import time

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr

from ocean4dvarnet.data import BaseDataModule, TrainingItem
from ocean4dvarnet.models import Lit4dVarNet, GradSolver


class LatentDecoderMR(torch.nn.Module):
    def __init__(self, dim_state, dim_latent, channel_dims,scale_factor,interp_mode='linear'):
        super().__init__()
        self.dim_state    = dim_state
        self.dim_latent   = dim_latent
        self.channel_dims = channel_dims
        self.scale_factor = scale_factor
        self.interp_mode  = interp_mode

        self.decode_residual = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=dim_state+dim_latent,
                out_channels=channel_dims,
                padding="same",
                kernel_size=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=channel_dims,
                out_channels=dim_state,
                padding="same",
                kernel_size=1,
            ),
        )

    def forward(self, batch):
        x = batch.input
        x = x.nan_to_num()

        x_up = torch.nn.functional.interpolate(x,scale_factor=self.scale_factor,mode=)
        dx   = self.decode_residual(x_up)

        return x_up[:,:self.dim_state,:,:] + dx 

class GradSolverWithLatent(GradSolver):
   """
    A gradient-based solver for optimization in 4D-VarNet.

    Attributes:
        prior_cost (nn.Module): The prior cost function.
        obs_cost (nn.Module): The observation cost function.
        grad_mod (nn.Module): The gradient modulation model.
        n_step (int): Number of optimization steps.
        lr_grad (float): Learning rate for gradient updates.
        lbd (float): Regularization parameter.
    """

    def __init__(self, prior_cost, obs_cost, grad_mod, latent_decoder, n_step, lr_grad=0.2, lbd=1.0, std_latent_init=0., **kwargs):
        """
        Initialize the GradSolver.

        Args:
            prior_cost (nn.Module): The prior cost function.
            obs_cost (nn.Module): The observation cost function.
            grad_mod (nn.Module): The gradient modulation model.
            n_step (int): Number of optimization steps.
            lr_grad (float, optional): Learning rate for gradient updates. Defaults to 0.2.
            lbd (float, optional): Regularization parameter. Defaults to 1.0.
        """
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, lbd=1.0,kwargs)

        self.latent_decoder  = latent_decoder
        self.std_latent_init = std_latent_init

    def init_state(self, batch, x_init=None):
        """
        Initialize the state for optimization.

        Args:
            batch (dict): Input batch containing data.
            x_init (torch.Tensor, optional): Initial state. Defaults to None.

        Returns:
            torch.Tensor: Initialized state.
        """
        if x_init is not None:
            return x_init

        # initialization using average-pooled obs inputs
        # for the coarse-scale component
        x = batch.input.nan_to_num().detach()
        m = torch.isnan( batch.input )
        
        x = nn.functional.avg_pool2d(x,self.latent_decoder.scale_factor)
        m = nn.functional.avg_pool2d(m.float(),self.latent_decoder.scale_factor)
        x = x / (m + 1e-8)

        # random initialisation for the latent representation
        size = [batch.input.shape[0], self.latent_decoder.dim_latent, *batch.input.shape[-2:]]
        latent_state_init = self.self.std_latent_init * torch.randn(size,device=batch.input.device)

        x_init_ = torch.cat( (x,latent_state_init) , dim = 1)

        return x_init_.detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        """
        Perform a single optimization step.

        Args:
            state (torch.Tensor): Current state.
            batch (dict): Input batch containing data.
            step (int): Current optimization step.

        Returns:
            torch.Tensor: Updated state.
        """
        var_cost = self.prior_cost(state) + self.lbd**2 * self.obs_cost(self.latent_decoder(state), batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
            + self.lr_grad * (step + 1) / self.n_step * grad
        )

        return state - state_update

    def forward(self, batch):
        """
        Perform the forward pass of the solver.

        Args:
            batch (dict): Input batch containing data.

        Returns:
            torch.Tensor: Final optimized state.
        """
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            #if not self.training:
            #    state = self.prior_cost.forward_ae(state)

                   
        return self.latent_decoder(state) # apply decoder from latent representation




