from collections import namedtuple
import functools as ft
import time

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr

from ocean4dvarnet.data import BaseDataModule, TrainingItem
from ocean4dvarnet.models import Lit4dVarNet,GradSolver

TrainingItemOSEwOSSE = namedtuple('TrainingItemOSEwOSSE', ['input', 'tgt','input_osse','tgt_osse','lon','lat'])
_LAT_TO_RAD = np.pi / 180.0


class ConvLstmGradModel(torch.nn.Module):
    """
    A convolutional LSTM model for gradient modulation.

    Attributes:
        dim_hidden (int): Number of hidden dimensions.
        gates (nn.Conv2d): Convolutional gates for LSTM.
        conv_out (nn.Conv2d): Output convolutional layer.
        dropout (nn.Dropout): Dropout layer.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None, bias=False):
        """
        Initialize the ConvLstmGradModel.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            downsamp (int, optional): Downsampling factor. Defaults to None.
        """
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def reset_state(self, inp):
        """
        Reset the internal state of the LSTM.

        Args:
            inp (torch.Tensor): Input tensor to determine state size.
        """
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        """
        Perform the forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x = x / self._grad_norm
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        return out


class ConvLstmGradModelUnet(torch.nn.Module):
    """
    A convolutional LSTM model for gradient modulation.

    Attributes:
        dim_hidden (int): Number of hidden dimensions.
        gates (nn.Conv2d): Convolutional gates for LSTM.
        conv_out (nn.Conv2d): Output convolutional layer.
        dropout (nn.Dropout): Dropout layer.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, dim_in, dim_hidden, unet=None, kernel_size=3, dropout=0.1, downsamp=None,bias=False):
        """
        Initialize the ConvLstmGradModel.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            downsamp (int, optional): Downsampling factor. Defaults to None.
        """
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias
        )

        if unet is not None:
            self.conv_out = unet
            self.use_unet = True
        else:
            self.use_unet = False
            self.conv_out = torch.nn.Conv2d(
                                dim_hidden, dim_in, kernel_size=kernel_size, 
                                padding=kernel_size // 2, bias=bias
                                )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def reset_state(self, inp):
        """
        Reset the internal state of the LSTM.

        Args:
            inp (torch.Tensor): Input tensor to determine state size.
        """
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        """
        Perform the forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x = x / self._grad_norm
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell

        if self.use_unet == True:
            out = self.conv_out.predict(hidden)
        else:
            out = self.conv_out(hidden)
            
        out = self.up(out)

        return out

class GradSolverZeroInit(GradSolver):
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

    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, lbd=1.0, **kwargs):
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
        super().__init__(prior_cost, obs_cost, grad_mod, n_step=n_step, lr_grad=lr_grad, lbd=lbd,**kwargs)

        self._grad_norm = None

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

        return torch.zeros_like(batch.input).detach().requires_grad_(True)

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
        var_cost = self.prior_cost(state) + self.lbd**2 * self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)

        state_update = (
             1. / self.n_step * gmod
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
        return state

class LatentDecoderMR(torch.nn.Module):
    def __init__(self, dim_state, dim_latent, channel_dims,scale_factor,interp_mode='linear',w_dx = None):
        super().__init__()
        self.dim_state    = dim_state
        self.dim_latent   = dim_latent
        self.channel_dims = channel_dims
        self.scale_factor = scale_factor
        self.interp_mode  = interp_mode
        if w_dx is not None :  self.w_dx =  w_dx 
        else: 
            self.w_dx = 1.

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

    def forward(self, x):
        x = x.nan_to_num()

        x_up = torch.nn.functional.interpolate(x,scale_factor=self.scale_factor,mode=self.interp_mode)
        dx   = self.decode_residual(x_up)

        return x_up[:,:self.dim_state,:,:] + self.w_dx * dx 


class LatentEncoderMR(torch.nn.Module):
    def __init__(self, dim_state, dim_latent, channel_dims,scale_factor):
        super().__init__()
        self.dim_state    = dim_state
        self.dim_latent   = dim_latent
        self.channel_dims = channel_dims
        self.scale_factor = scale_factor

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=dim_state,
                out_channels=2*channel_dims,
                padding="same",
                kernel_size=3,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=2*channel_dims,
                out_channels=channel_dims,
                padding="same",
                kernel_size=3,
            ),
            torch.nn.AvgPool2d((scale_factor,scale_factor)),
            torch.nn.Conv2d(
                in_channels=channel_dims,
                out_channels=2*channel_dims,
                padding="same",
                kernel_size=3,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=2*channel_dims,
                out_channels=dim_latent,
                padding="same",
                kernel_size=3,
            ),
        )

    def forward(self, x):
        x = x.nan_to_num()
        
        dx_latent = self.encoder(x)
        x_latent  = torch.nn.functional.avg_pool2d(x,self.scale_factor)

        return torch.cat((x_latent,dx_latent),dim=1) 


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

    def __init__(self, prior_cost, obs_cost, grad_mod, latent_decoder, latent_encoder, n_step, lr_grad=0.2, lbd=1.0, std_latent_init=0., **kwargs):
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
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad, lbd,**kwargs)

        self.latent_decoder  = latent_decoder
        self.latent_encoder  = latent_encoder
        self.std_latent_init = torch.nn.Parameter(torch.Tensor([std_latent_init]),requires_grad=True)

    def init_latent_from_state(self,x):
        # initialization using average-pooled obs inputs
        # for the coarse-scale component
        x = x.nan_to_num().detach()
        m = 1. - torch.isnan( x ).float()
        
        x = torch.nn.functional.avg_pool2d(x,self.latent_decoder.scale_factor)
        m = torch.nn.functional.avg_pool2d(m.float(),self.latent_decoder.scale_factor)
        x = x / (m + 1e-8)

        # random initialisation for the latent representation
        size = [x.shape[0], self.latent_decoder.dim_latent, *x.shape[-2:]]
        latent_state_init = self.std_latent_init * torch.randn(size,device=x.device)

        return torch.cat( (x,latent_state_init) , dim = 1)

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
        x_init_ = self.init_latent_from_state( batch.input)

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
            self.grad_mod.reset_state(state) #batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            #if not self.training:
            #    state = self.prior_cost.forward_ae(state)

        #print(self.latent_decoder(state).shape)

        return self.latent_decoder(state),state # apply decoder from latent representation

class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def __init__(self,  
                 w_mse,w_grad_mse, w_mse_lr, w_grad_mse_lr, w_prior,
                 *args, **kwargs):
        _val_rec_weight = kwargs.pop(
            "val_rec_weight",
            kwargs["rec_weight"],
        )
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "val_rec_weight",
            torch.from_numpy(_val_rec_weight),
            persistent=False,
        )

        self._n_rejected_batches = 0

        self.w_mse = w_mse
        self.w_grad_mse = w_grad_mse
        self.w_mse_lr = w_mse_lr
        self.w_grad_mse_lr = w_grad_mse_lr
        self.w_prior = w_prior
    
    def get_rec_weight(self, phase):
        rec_weight = self.rec_weight
        if phase == "val":
            rec_weight = self.val_rec_weight
        return rec_weight

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if loss is None:
            self._n_rejected_batches += 1
        return loss

    def on_train_epoch_end(self):
        self.log(
            "n_rejected_batches",
            self._n_rejected_batches,
            on_step=False,
            on_epoch=True,
        )

    def loss_mse(self,batch,out,phase):
        loss =  self.weighted_mse(out - batch.tgt,
            self.get_rec_weight(phase),
        )

        grad_loss =  self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        return loss, grad_loss

    def loss_prior(self,batch,out,phase):

        # prior cost for estimated latent state    
        loss_prior_out = self.solver.prior_cost(out) # Why using init_state

        # prior cost for true state
        loss_prior_tgt = self.solver.prior_cost( batch.tgt.nan_to_num() )

        return loss_prior_out,loss_prior_tgt

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        loss, out = self.base_step(batch, phase)

        loss_mse = self.loss_mse(batch,out,phase)
        loss_prior = self.loss_prior(batch,out.detach(),phase)

        training_loss = self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1]
        training_loss += self.w_prior * loss_prior[0] + self.w_prior * loss_prior[1]

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss_mse[0] * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                training_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

            self.log(
                f"{phase}_gloss",
                loss_mse[1],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_ploss_out",
                loss_prior[0],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_ploss_gt",
                loss_prior[1],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        return loss, out

def cosanneal_lr_adam_twosolvers(lit_mod, lr, T_max=100, weight_decay=0.):
    """
    Configure an Adam optimizer with cosine annealing learning rate scheduling.

    Args:
        lit_mod: The Lightning module containing the model.
        lr (float): The base learning rate.
        T_max (int): Maximum number of iterations for the scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Returns:
        dict: A dictionary containing the optimizer and scheduler.
    """
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
            {"params": lit_mod.solver2.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver2.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver2.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max
        ),
    }

class Lit4dVarNetTwoSolvers(Lit4dVarNetIgnoreNaN):
    def __init__(self,
                 solver2,  
                 w_mse,w_grad_mse, w_mse_lr, w_grad_mse_lr, w_prior,
                 scale_solver,
                 w_solver2=None,
                 *args, **kwargs):

        super().__init__(w_mse,w_grad_mse, w_mse_lr, w_grad_mse_lr, w_prior,*args, **kwargs)

        self.solver2 = solver2
        self.scale_solver = scale_solver
        if w_solver2 is not None:
            self.w_solver2 = w_solver2
        else:
            self.w_solver2 = 0.5

    def loss_mse_lr(self,batch,out,phase,scale=2.):
        # compute mse losses for average-pooled state
        m = 1. - torch.isnan( batch.tgt ).float()
        
        tgt_lr   = torch.nn.functional.avg_pool2d(batch.tgt,scale)
        m = torch.nn.functional.avg_pool2d(m.float(),scale)
        tgt_lr = tgt_lr / (m + 1e-8)    
        
        out_lr = torch.nn.functional.avg_pool2d(out,scale)

        wrec = self.get_rec_weight(phase)
        wrec_lr = torch.nn.functional.avg_pool2d(wrec.view(1,wrec.shape[0],wrec.shape[1],wrec.shape[2]),scale)
        wrec_lr = wrec_lr.squeeze()

        loss =  self.weighted_mse( m * ( out_lr - tgt_lr) ,
            wrec_lr,
        )

        grad_loss =  self.weighted_mse(
            m * ( kfilts.sobel(out_lr) - kfilts.sobel(tgt_lr) ),
            wrec_lr,
        )

        return loss, grad_loss


    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        loss, out1, out2 = self.base_step(batch, phase)

        loss_mse_1 = self.loss_mse_lr(batch,out1,phase,scale=self.scale_solver)
        loss_prior_1 = self.loss_prior(batch,out1.detach(),phase)

        loss_mse_2 = self.loss_mse(batch,out2,phase)
        loss_prior_2 = self.loss_prior(batch,out2.detach(),phase)

        training_loss = self.w_mse * loss_mse_1[0] + self.w_grad_mse * loss_mse_1[1]
        training_loss += self.w_prior * loss_prior_1[0] + self.w_prior * loss_prior_1[1]

        training_loss_2 = self.w_mse * loss_mse_2[0] + self.w_grad_mse * loss_mse_2[1]
        training_loss_2 += self.w_prior * loss_prior_2[0] + self.w_prior * loss_prior_2[1]
        training_loss = (1. - self.w_solver2) * training_loss + self.w_solver2 * training_loss_2

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * ( (1. - self.w_solver2) * loss_mse_1[0] + self.w_solver2 * loss_mse_2[0]) * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                training_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

            self.log(
                f"{phase}_gloss",
                (1. - self.w_solver2) * loss_mse_1[1] + self.w_solver2 * loss_mse_2[1],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_ploss_out",
                loss_prior_2[0],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_ploss_gt",
                loss_prior_2[1],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

        return training_loss, out2

    def base_step(self, batch, phase):
        
        # apply first solver 
        m = 1. - torch.isnan( batch.input ).float()
        input_lr   = torch.nn.functional.avg_pool2d(batch.input.nan_to_num(),self.scale_solver)
        m = torch.nn.functional.avg_pool2d(m.float(),self.scale_solver)
        input_lr = input_lr / (m + 1e-8)  
        out1 = self.solver(batch= TrainingItem(input_lr.detach(), None))
        out1 = torch.nn.functional.interpolate(out1,scale_factor=self.scale_solver,mode='bilinear')

        with torch.set_grad_enabled(True):
             # apply 2nd solver
            out2 = self.solver2.init_state(None, 1. * out1.detach())
            out2 = out2.requires_grad_(True)

            self.solver2.grad_mod.reset_state(out2)
            for step in range(self.solver2.n_step):
                out2 = self.solver2.solver_step(out2, batch, step=step)
                if not self.training:
                    out2 = out2.detach().requires_grad_(True)

        loss = self.weighted_mse(out2 - batch.tgt, self.get_rec_weight(phase))

        return loss, out2, out1


# Utils
# -----


def load_glorys12_data(tgt_path, inp_path, tgt_var="zos", inp_var="input"):
    isel = None  # dict(time=slice(-465, -265))

    _start = time.time()

    
    print('..... Start loading dataset',flush=True)
    
    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
    )
    inp = xr.open_dataset(inp_path)[inp_var].isel(isel)

    ds = (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values)),
            inp.coords,
        )
        .to_array()
        .sortby("variable")
    )

    print(f">>> Durée de chargement : {time.time() - _start:.4f} s",flush=True)
    return ds

def load_glorys12_data_on_fly_inp(
    tgt_path,
    inp_path,
    tgt_var="zos",
    inp_var="input",
):
 
    print('..... Start lazy loading',flush=True)

    isel = None  # dict(time=slice(-365 * 2, None))

    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
        #.rename(latitude="lat", longitude="lon")
    )

    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .isel(isel)
        #.rename(latitude="lat", longitude="lon")
    )
    return tgt, inp


def train(trainer, dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    start = time.time()
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    print(f"Durée d'apprentissage : {time.time() - start:.3} s")


class Lit4dVarNetIgnoreNaNLatent(Lit4dVarNetIgnoreNaN):
    def __init__(self, w_latent_ae, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_latent_ae = w_latent_ae

    def forward(self, batch):
        """
        Forward pass through the solver.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: first output of the LatentSolver.
        """
        return self.solver(batch)[0]
    
    def base_step(self, batch, phase):
        out, latent = self.solver(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        return loss, out, latent

    def loss_mse(self,batch,out,phase):
        loss =  self.weighted_mse(out - batch.tgt,
            self.get_rec_weight(phase),
        )

        grad_loss =  self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        return loss, grad_loss

    def loss_mse_lr(self,batch,out,phase,scale=2.):
        # compute mse losses for average-pooled state
        m = 1. - torch.isnan( batch.tgt ).float()
        
        tgt_lr   = torch.nn.functional.avg_pool2d(batch.tgt,scale)
        m = torch.nn.functional.avg_pool2d(m.float(),scale)
        tgt_lr = tgt_lr / (m + 1e-8)    
        
        out_lr = torch.nn.functional.avg_pool2d(out,scale)

        wrec = self.get_rec_weight(phase)
        wrec_lr = torch.nn.functional.avg_pool2d(wrec.view(1,wrec.shape[0],wrec.shape[1],wrec.shape[2]),scale)
        wrec_lr = wrec_lr.squeeze()

        loss =  self.weighted_mse( m * ( out_lr - tgt_lr) ,
            wrec_lr,
        )

        grad_loss =  self.weighted_mse(
            m * ( kfilts.sobel(out_lr) - kfilts.sobel(tgt_lr) ),
            wrec_lr,
        )

        return loss, grad_loss
    
    def loss_prior(self,batch,latent,phase):

        # prior cost for estimated latent state    
        loss_prior_out = self.solver.prior_cost(latent) # Why using init_state

        # prior cost for true state
        latent_tgt = self.solver.init_latent_from_state( batch.tgt )
        latent_tgt = torch.cat( (latent_tgt[:,:self.solver.latent_decoder.dim_state,:,:],
                                 latent[:,self.solver.latent_decoder.dim_state:,:,:]) , dim = 1)
        loss_prior_tgt = self.solver.prior_cost(latent_tgt.detach()) # Why using init_state

        return loss_prior_out,loss_prior_tgt

    def loss_latent_ae(self,batch,out,phase):

        # prior cost for estimated latent state  
        enc = self.solver.latent_encoder( batch.tgt.nan_to_num() )
        dec = self.solver.latent_decoder( enc )
        loss_ae = torch.mean( (dec - batch.tgt.nan_to_num() )**2   )

        enc = self.solver.latent_encoder( out )
        dec = self.solver.latent_decoder( enc )
     
        loss_ae += torch.mean( (dec - out )**2   )

        return loss_ae

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        loss, out, latent = self.base_step(batch, phase)

        # training losses
        loss_mse_hr = self.loss_mse(batch,out,phase)
        loss_mse_lr = self.loss_mse_lr(batch,out,phase,scale=self.solver.latent_decoder.scale_factor)

        loss_prior = self.loss_prior(batch,latent.detach(),phase)

        loss_latent_ae = self.loss_latent_ae(batch,out.detach(),phase)

        training_loss = self.w_mse * loss_mse_hr[0] + self.w_grad_mse * loss_mse_hr[1] 
        training_loss += self.w_mse_lr * loss_mse_lr[0] + self.w_grad_mse_lr * loss_mse_lr[1]
        training_loss += self.w_prior * loss_prior[0] + self.w_prior * loss_prior[1]
        training_loss += self.w_latent_ae * loss_latent_ae 

        # log
        self.log(
            f"{phase}_mse",
            10000 * loss_mse_hr[0] * self.norm_stats[phase][1] ** 2,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_loss",
            training_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_gloss",
            loss_mse_hr[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_loss_lr",
            loss_mse_lr[0],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_gloss_lr",
            loss_mse_lr[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_ploss_out",
            loss_prior[0],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_ploss_gt",
            loss_prior[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        
        return training_loss, out

class UnetSolver(torch.nn.Module):
    def __init__(self, dim_in, channel_dims, max_depth=None,bias=True):
        super().__init__()

        if max_depth is not None :
            self.max_depth = np.max( max_depth , len(channel_dims) // 3 )
        else: 
            self.max_depth = len(channel_dims) // 3
        
        self.ups = torch.nn.ModuleList()
        self.up_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.residues = list()

        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias
            ),
            torch.nn.ReLU(),
        )

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=dim_in,
                padding="same",
                kernel_size=3,
                bias=bias
            )
        )

        self.final_linear = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_in))

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.up_pools.append(
                torch.nn.ConvTranspose2d(
                    in_channels=channel_dims[depth * 3 + 3],
                    out_channels=channel_dims[depth * 3 + 2],
                    kernel_size=2,
                    stride=2,
                    bias=bias
                )
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.down_pools.append(torch.nn.MaxPool2d(kernel_size=2))

    def unet_step(self, x, depth):
        x, residue = self.down(x, depth)
        self.residues.append(residue)

        if depth == self.max_depth - 1:
            x = self.bottom_transform(x)
        else:
            x = self.unet_step(x, depth + 1)

        return self.up(x, depth)

    def forward(self, batch):
        x = batch.input
        x = x.nan_to_num()
 #       x = self.final_up(self.unet_step(x, depth=0))
 #       x = torch.permute(x, dims=(0, 2, 3, 1))
 #       x = self.final_linear(x)
 #       x = torch.permute(x, dims=(0, 3, 1, 2))
        return self.predict(x)

    def predict(self,x):
        x = self.final_up(self.unet_step(x, depth=0))
        x = torch.permute(x, dims=(0, 2, 3, 1))
        x = self.final_linear(x)
        x = torch.permute(x, dims=(0, 3, 1, 2))
        return x        

    def down(self, x, depth):
        x = self.downs[depth](x)
        return self.down_pools[depth](x), x

    def up(self, x, depth):
        x = self.up_pools[depth](x)
        x = self.concat_residue(x)
        return self.ups[depth](x)

    def concat_residue(self, x):
        if len(self.residues) != 0:
            residue = self.residues.pop(-1)

            _, _, h_x, w_x = x.shape
            _, _, h_r, w_r = residue.shape

            pad_h = h_r - h_x
            pad_w = w_r - w_x

            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect", value=0)

            return torch.concat((x, residue), dim=1)
        else:
            return x


class BilinAEPriorCostNoBias(torch.nn.Module):
    """
    A prior cost model using bilinear autoencoders.

    Attributes:
        bilin_quad (bool): Whether to use bilinear quadratic terms.
        conv_in (nn.Conv2d): Convolutional layer for input.
        conv_hidden (nn.Conv2d): Convolutional layer for hidden states.
        bilin_1 (nn.Conv2d): Bilinear layer 1.
        bilin_21 (nn.Conv2d): Bilinear layer 2 (part 1).
        bilin_22 (nn.Conv2d): Bilinear layer 2 (part 2).
        conv_out (nn.Conv2d): Convolutional layer for output.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True):
        """
        Initialize the BilinAEPriorCost module.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            downsamp (int, optional): Downsampling factor. Defaults to None.
            bilin_quad (bool, optional): Whether to use bilinear quadratic terms. Defaults to True.
        """
        super().__init__()
        self.bilin_quad = bilin_quad
        self.conv_in = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2 , bias = False
        )

        self.bilin_1 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2 , bias = False
        )
        self.bilin_21 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2 , bias = False
        )
        self.bilin_22 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2 , bias = False
        )

        self.conv_out = torch.nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2 , bias = False
        )

        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def forward_ae(self, x):
        """
        Perform the forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the autoencoder.
        """
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = (
            self.bilin_21(x)**2
            if self.bilin_quad
            else (self.bilin_21(x) * self.bilin_22(x))
        )
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state):
        """
        Compute the prior cost using the autoencoder.

        Args:
            state (torch.Tensor): The current state tensor.

        Returns:
            torch.Tensor: The computed prior cost.
        """
        return F.mse_loss(state, self.forward_ae(state))


class BilinAEPriorCostTwoScale(torch.nn.Module):
    """
    A prior cost model using bilinear autoencoders.

    Attributes:
        bilin_quad (bool): Whether to use bilinear quadratic terms.
        conv_in (nn.Conv2d): Convolutional layer for input.
        conv_hidden (nn.Conv2d): Convolutional layer for hidden states.
        bilin_1 (nn.Conv2d): Bilinear layer 1.
        bilin_21 (nn.Conv2d): Bilinear layer 2 (part 1).
        bilin_22 (nn.Conv2d): Bilinear layer 2 (part 2).
        conv_out (nn.Conv2d): Convolutional layer for output.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True, bias=True):
        """
        Initialize the BilinAEPriorCost module.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            downsamp (int, optional): Downsampling factor. Defaults to None.
            bilin_quad (bool, optional): Whether to use bilinear quadratic terms. Defaults to True.
        """
        super().__init__()
        self.bilin_quad = bilin_quad
        self.conv_in = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.conv_hidden = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.bilin_1 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.bilin_21 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.bilin_22 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.conv_out = torch.nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )


        self.conv_in_lr = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.conv_hidden_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.bilin_1_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.bilin_21_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.bilin_22_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.conv_out_lr = torch.nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )


        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def forward_ae(self, x):
        """
        Perform the forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the autoencoder.
        """

        # coarse-scale processing
        x_ = self.down(x)
        x_ = self.conv_in_lr(x_)
        x_ = self.conv_hidden_lr(torch.nn.functional.relu(x_))

        nonlin = (
            self.bilin_21_lr(x_)**2
            if self.bilin_quad
            else (self.bilin_21_lr(x_) * self.bilin_22_lr(x_))
        )

        x_ = self.conv_out_lr(
            torch.cat([self.bilin_1_lr(x_), nonlin], dim=1)
        )
        dx = self.up(x_)

        # fine-scale processing
        x = self.conv_in(x)
        x = self.conv_hidden(torch.nn.functional.relu(x))

        nonlin = (
            self.bilin_21(x)**2
            if self.bilin_quad
            else (self.bilin_21(x) * self.bilin_22(x))
        )
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        
        return x + dx

    def forward(self, state):
        """
        Compute the prior cost using the autoencoder.

        Args:
            state (torch.Tensor): The current state tensor.

        Returns:
            torch.Tensor: The computed prior cost.
        """
        return torch.nn.functional.mse_loss(state, self.forward_ae(state))


class UnetSolver2(UnetSolver):
    def __init__(self, dim_in, channel_dims, max_depth=None,dim_out=None,bias=True):
        super().__init__(dim_in, channel_dims, max_depth)

        if dim_out is None :
            dim_out = dim_in

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4*dim_out,
                padding="same",
                kernel_size=3,
                bias=bias
            ) )

        self.final_linear = torch.nn.Sequential(torch.nn.Linear(4*dim_out, dim_out))


class UpsampleWInterpolate(torch.nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None, interp_mode='bilinear',bias=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interp_mode = interp_mode
        if use_conv:
            self.conv  = torch.nn.Conv2d(in_channels=channels,out_channels=out_channels,
                                        padding="same",kernel_size=1,bias=bias)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode=self.interp_mode)
        if self.use_conv:
            x = self.conv(x)
        return x

class UnetSolverBilin(UnetSolver2):
    def __init__(self, dim_in, channel_dims, max_depth=None,dim_out=None,interp_mode='bilinear',dropout=0.1,activation_layer=torch.nn.ReLU(),bias=True):
        super().__init__(dim_in, channel_dims, max_depth=max_depth,bias=bias)

        if dim_out is None :
            dim_out = dim_in

        self.up_pools   = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()

        self.interp_mode = interp_mode
        self.dropout = dropout
        
        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            activation_layer,
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            activation_layer,
        )

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=dim_in,
                padding="same",
                kernel_size=3,
                bias=bias,
            )
        )

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                )
            )
            self.up_pools.append(
                    UpsampleWInterpolate(channels=channel_dims[depth * 3 + 3], use_conv=True, 
                                        out_channels=channel_dims[depth * 3 + 2], interp_mode= self.interp_mode)
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                )
            )

            self.down_pools.append(torch.nn.AvgPool2d(kernel_size=2))

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4*dim_out,
                padding="same",
                kernel_size=3,
                bias=bias,
            ) )
        self.final_linear = torch.nn.Sequential(torch.nn.Linear(4*dim_out, dim_out,bias=bias))


class UnetSolverwithLonLat(UnetSolver2):
    def __init__(self, dim_in, channel_dims, max_depth=None,dim_out=None):
        super().__init__(dim_in+3, channel_dims, max_depth,dim_out)

    def forward(self, batch):
        # pre-process lon,lat 
        lat = _LAT_TO_RAD * batch.lat.view(-1,1,batch.lat.shape[1],1).repeat(1,1,1,batch.input.shape[-1])
        lon = _LAT_TO_RAD * batch.lon.view(-1,1,1,batch.lon.shape[1]).repeat(1,1,batch.input.shape[2],1)

        x_lon_lat = torch.cat( (batch.input.nan_to_num(),torch.cos(lat),torch.cos(lon),torch.sin(lon)),dim=1)

        return self.predict(x_lon_lat)

class UnetSolverwithGAttn(UnetSolver2):
    def __init__(self, dim_in, channel_dims, dim_inp_attn=None, max_depth=None,dim_out=None):
        super().__init__(dim_out, channel_dims, max_depth)

        print()
        if dim_out is None :
            self.dim_out = dim_in
        else: 
            self.dim_out = dim_out
       
        if dim_inp_attn is None :
            self.dim_inp_attn = dim_in
        else:
            self.dim_inp_attn = dim_inp_attn

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4*dim_out,
                padding="same",
                kernel_size=3,
            ) )
        self.final_linear = torch.nn.Sequential(torch.nn.Linear(4*dim_out, dim_out))


        self.global_attn = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=dim_inp_attn,
                out_channels=32,
                padding="same",
                kernel_size=3,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=4*dim_out,
                padding="same",
                kernel_size=3,
            ),
            torch.nn.Softmax(dim=1),
        )
    def predict(self,x):
        x_ = x[:,:self.dim_out,:,:]
        x_w = x[:,x.shape[1]-self.dim_inp_attn:,:,:]

        x = self.final_up(self.unet_step(x_, depth=0))
        w = self.global_attn(x_w)

        x = x * w

        x = torch.permute(x, dims=(0, 2, 3, 1))
        x = self.final_linear(x)
        x = torch.permute(x, dims=(0, 3, 1, 2))
        return x        

class UnetSolverWithPrepro(UnetSolver2):
    def __init__(self, dim_in, channel_dims, max_depth=None,dim_out=None,kernel_prepro=None):

        if kernel_prepro is not None:
            self.kernel_prepro = kernel_prepro
        else:
            self.kernel_prepro = (dim_in,)

        super().__init__(dim_in, channel_dims, max_depth,dim_out)

    def preprocess_input_data(self, batch):
        inp = batch.input.nan_to_num().view(-1,1,batch.input.shape[1],batch.input.shape[2], batch.input.shape[3])
        mask = 1. - batch.input.isnan().float().view(-1,1,batch.input.shape[1],batch.input.shape[2], batch.input.shape[3])

        new_inp = None
        for kernel_size in self.kernel_prepro:
            inp_avg = torch.nn.functional.avg_pool3d(inp, (kernel_size,1,1))
            m_avg = torch.nn.functional.avg_pool3d(mask, (kernel_size,1,1))
            inp_avg = inp_avg / ( m_avg + 1e-8 )
            inp_avg = inp_avg.view(batch.input.shape[0],-1,batch.input.shape[2], batch.input.shape[3])

            if new_inp is not None:
                new_inp = torch.cat((new_inp, inp_avg), dim=1)
            else:
                new_inp = inp_avg

        #print(f"new_inp shape: {new_inp.shape}, inp shape: {inp.shape}",flush=True)
        new_inp = torch.cat((batch.input.nan_to_num(), new_inp), dim=1)

        return new_inp
    
    def forward(self, batch):
        x_inp_prepro = self.preprocess_input_data(batch)

        # pre-process lon,lat 
        #lat = _LAT_TO_RAD * batch.lat.view(-1,1,batch.lat.shape[1],1).repeat(1,1,1,batch.input.shape[-1])
        #lon = _LAT_TO_RAD * batch.lon.view(-1,1,1,batch.lon.shape[1]).repeat(1,1,batch.input.shape[2],1)

        #x_lon_lat = torch.cat( (batch.input.nan_to_num(),torch.cos(lat),torch.cos(lon),torch.sin(lon)),dim=1)

        return self.predict(x_inp_prepro)



def cosanneal_lr_adam_base(lit_mod, lr, T_max=100, weight_decay=0.):
    """
    Configure an Adam optimizer with cosine annealing learning rate scheduling.

    Args:
        lit_mod: The Lightning module containing the model.
        lr (float): The base learning rate.
        T_max (int): Maximum number of iterations for the scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Returns:
        dict: A dictionary containing the optimizer and scheduler.
    """
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.parameters(), "lr": lr},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max
        ),
    }

class LitUnetFromLit4dVarNetIgnoreNaN(Lit4dVarNetIgnoreNaN):
    def __init__(self,  
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_mse(self,batch,out,phase):
        loss =  self.weighted_mse(out - batch.tgt,
            self.get_rec_weight(phase),
        )

        grad_loss =  self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        return loss, grad_loss

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        self.log(
            f"{phase}_gloss",
            grad_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        loss_mse = self.loss_mse(batch,out,phase)
        training_loss = self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1] 

        # log
        self.log(
            f"{phase}_gloss",
            loss_mse[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

            if phase == "val":
                # Log the loss in Gulfstream
                loss_gf = self.weighted_mse(
                    out[:, :, 445:485, 420:460].detach().cpu().data
                    - batch.tgt[:, :, 445:485, 420:460].detach().cpu().data,
                    np.ones_like(out[:, :, 445:485, 420:460].detach().cpu().data),
                )
                self.log(
                    f"{phase}_loss_gulfstream",
                    loss_gf,
                    on_step=False,
                    on_epoch=True,
                )

        return loss, out

class LitUnetFromLit4dVarNetWithInit(LitUnetFromLit4dVarNetIgnoreNaN):
    def __init__(self, init_state=None, scale_init_state = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_state = init_state
        if scale_init_state is not None :
            self.scale_init_state = scale_init_state
        else:
            self.scale_init_state = 4

    def loss_mse(self,batch,out,phase):
        loss =  self.weighted_mse(out - batch.tgt,
            self.get_rec_weight(phase),
        )

        grad_loss =  self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        return loss, grad_loss

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        self.log(
            f"{phase}_gloss",
            grad_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        loss_mse = self.loss_mse(batch,out,phase)
        training_loss = self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1] 

        # log
        self.log(
            f"{phase}_gloss",
            loss_mse[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        return training_loss, out

    def compute_init_state(self, x, scale):

        if self.init_state == 'zeros' :
            return torch.zeros_like(x)
        else:
            # initialization using average-pooled obs inputs
            # for the coarse-scale component
            x = x.nan_to_num().detach()
            m = 1. - torch.isnan( x ).float()        

            x_ = torch.nn.functional.avg_pool2d(x,scale)
            m = torch.nn.functional.avg_pool2d(m.float(),scale)
            x_ = x_ / (m + 1e-8)

            # time average
            x_ =( torch.mean(x_, dim=1, keepdim=True) ).repeat(1,x.shape[1],1,1) 

            # reinterpolate
            x_ = torch.nn.functional.interpolate(x_,scale_factor=scale,mode='bilinear')

            return x_.detach()

    def base_step(self, batch, phase):

        out_init = self.compute_init_state(batch.input, self.scale_init_state)
        out = out_init + self(batch=batch)

        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

            if phase == "val":
                # Log the loss in Gulfstream
                loss_gf = self.weighted_mse(
                    out[:, :, 445:485, 420:460].detach().cpu().data
                    - batch.tgt[:, :, 445:485, 420:460].detach().cpu().data,
                    np.ones_like(out[:, :, 445:485, 420:460].detach().cpu().data),
                )
                self.log(
                    f"{phase}_loss_gulfstream",
                    loss_gf,
                    on_step=False,
                    on_epoch=True,
                )

        return loss, out


class LitUnetWithLonLat(LitUnetFromLit4dVarNetIgnoreNaN):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def base_step(self, batch, phase):
        out = self.solver(batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

            if phase == "val":
                # Log the loss in Gulfstream
                loss_gf = self.weighted_mse(
                    out[:, :, 445:485, 420:460].detach().cpu().data
                    - batch.tgt[:, :, 445:485, 420:460].detach().cpu().data,
                    np.ones_like(out[:, :, 445:485, 420:460].detach().cpu().data),
                )
                self.log(
                    f"{phase}_loss_gulfstream",
                    loss_gf,
                    on_step=False,
                    on_epoch=True,
                )

        return loss, out


class LitUnetOSEwOSSE(LitUnetFromLit4dVarNetIgnoreNaN):
    def __init__(self,  w_ose, w_osse, scale_loss_ose, osse_type, sig_noise_ose2osse, 
                 patch_normalization=None, normalization_noise=0.,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.w_ose = w_ose
        self.w_osse = w_osse
        self.scale_loss_ose = scale_loss_ose
        self.osse_type = osse_type
        self.sig_noise_ose2osse = sig_noise_ose2osse

        self.patch_normalization = patch_normalization
        self.normalization_noise = normalization_noise


    def aug_data_with_ose2osse_noise(self,batch,sig_noise_ose2osse=1,osse_type='keep-original'):

        if osse_type == 'noise-from-ose':
            noise_ose = batch.input - batch.tgt
            noise_ose = noise_ose[torch.randperm(noise_ose.size(0)),:,:,:]
            
            #print('\n ... noise mean: ', torch.nanmean(noise_ose).detach().cpu().numpy(),
            #      '  std: ',torch.sqrt(torch.nanmean( (noise_ose  - torch.nanmean(noise_ose))**2 )).detach().cpu().numpy())

            scale_noise = torch.rand((noise_ose.shape[0],)).to(device=batch.input.device)
            scale_noise = sig_noise_ose2osse * scale_noise.view(-1,1,1,1).repeat(1,noise_ose.shape[1],noise_ose.shape[2],noise_ose.shape[3])

            input_osse_tgt_from_ose = batch.tgt_osse + scale_noise * noise_ose

            return TrainingItemOSEwOSSE(batch.input, batch.tgt,
                                        input_osse_tgt_from_ose, batch.tgt_osse,
                                        batch.lon, batch.lat)

        return batch

    def apply_patch_normalization(self, batch, phase):
        #patch_normalization = 'from-obs'#None # #'from-gt-ose' # 
        #normalization_noise = True #False

        if self.patch_normalization == 'from-obs' :
            m_new = torch.nanmean( batch.input , dim=(1,2,3) )
            m_new = m_new.view(-1,1,1,1).repeat(1,batch.input.shape[1],batch.input.shape[2],batch.input.shape[3])

            s_new = torch.sqrt( torch.nanmean( (batch.input - m_new )**2 , dim=(1,2,3) ) )
            s_new = s_new.view(-1,1,1,1).repeat(1,batch.input.shape[1],batch.input.shape[2],batch.input.shape[3])
        elif self.patch_normalization == 'from-gt-ose' :
            m_new = torch.nanmean( batch.tgt , dim=(1,2,3) )
            m_new = m_new.view(-1,1,1,1).repeat(1,batch.input.shape[1],batch.input.shape[2],batch.input.shape[3])

            s_new = torch.sqrt( torch.nanmean( (batch.tgt - m_new )**2 , dim=(1,2,3) ) )
            s_new = s_new.view(-1,1,1,1).repeat(1,batch.input.shape[1],batch.input.shape[2],batch.input.shape[3])
        else:
            m_new = torch.zeros_like(batch.input)
            s_new = torch.ones_like(batch.input)

        if ( self.normalization_noise > 0 ) & (phase == 'train') :
            m_noise = self.normalization_noise * torch.randn( (batch.input.shape[0],1,1,1) , device=batch.input.device )
            s_noise = 1. + self.normalization_noise * ( torch.rand( (batch.input.shape[0],1,1,1), device=batch.input.device ) - 0.5 )

            m_noise = m_noise.repeat(1,batch.input.shape[1],batch.input.shape[2],batch.input.shape[3])
            s_noise = s_noise.repeat(1,batch.input.shape[1],batch.input.shape[2],batch.input.shape[3])


            m_new = m_new + m_noise
            s_new = s_new * s_noise

        return TrainingItemOSEwOSSE((batch.input- m_new) / s_new ,
                                      (batch.tgt - m_new) / s_new,
                                      None, None,
                                      batch.lon,
                                      batch.lat), m_new, s_new
  

    def base_step(self, batch, phase):
        # patch-based normalisation for OSE patches
        batch_,m_new, s_new = self.apply_patch_normalization(batch,phase)
      
        # apply solver to ose patches
        out_ose = self.solver(batch_)
        out_ose = (out_ose * s_new) + m_new

        # sampling OSSE input data
        batch = self.aug_data_with_ose2osse_noise(batch,
                                                  sig_noise_ose2osse=self.sig_noise_ose2osse,
                                                  osse_type=self.osse_type)

        batch_osse = TrainingItemOSEwOSSE(batch.input_osse,
                                          batch.tgt_osse,
                                          None, None,
                                          batch.lon,
                                          batch.lat)

        # apply solver to ose patches
        out_osse = self.solver(batch_osse)

        return out_ose, out_osse, batch_osse

    def loss_mse_lr(self,batch,out,phase,scale=2.):
        # compute mse losses for average-pooled state
        m = 1. - torch.isnan( batch.tgt ).float()
        
        tgt_lr   = torch.nn.functional.avg_pool2d(batch.tgt,scale)
        m = torch.nn.functional.avg_pool2d(m.float(),scale)
        tgt_lr = tgt_lr / (m + 1e-8)    
        
        out_lr = torch.nn.functional.avg_pool2d(out,scale)
        out_lr = out_lr / (m + 1e-8)    

        wrec = self.get_rec_weight(phase)
        wrec_lr = torch.nn.functional.avg_pool2d(wrec.view(1,wrec.shape[0],wrec.shape[1],wrec.shape[2]),scale)
        wrec_lr = wrec_lr.squeeze()

        loss =  self.weighted_mse( m * ( out_lr - tgt_lr) ,
            wrec_lr,
        )

        grad_loss =  self.weighted_mse(
            m * ( kfilts.sobel(out_lr) - kfilts.sobel(tgt_lr) ),
            wrec_lr,
        )

        return loss, grad_loss
    
    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:

            #print( batch.tgt.isfinite().float().mean() )
            #print( batch.input.isfinite().float().mean() )
            #print('\n ****')
            return None, None

        out_ose, out_osse, batch_osse = self.base_step(batch, phase)

        # training losses
        loss_mse_ose = self.loss_mse_lr(batch,out_ose,phase,scale=self.scale_loss_ose)
        loss_mse_osse = self.loss_mse(batch_osse,out_osse,phase)

        #print('\n')
        #print('\n')
        #print(loss_mse_ose[0].detach().cpu().numpy(), loss_mse_ose[1].detach().cpu().numpy(),
        #      loss_mse_osse[0].detach().cpu().numpy(), loss_mse_osse[1].detach().cpu().numpy(),
        #      flush=True)
        #print( torch.sqrt( torch.nanmean( (batch.tgt - batch.input)**2 ) ).detach().cpu().numpy(),
        #       torch.sqrt( torch.nanmean( (batch.tgt_osse - batch.input_osse)**2 ) ).detach().cpu().numpy(),
        #       flush=True)

        training_loss = self.w_ose * ( self.w_mse * loss_mse_ose[0] + self.w_grad_mse * loss_mse_osse[1] )
        training_loss += self.w_osse * ( self.w_mse * loss_mse_osse[0] + self.w_grad_mse * loss_mse_osse[1] )

        self.log(
            f"{phase}_gloss",
            loss_mse_osse[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_mse",
            10000 * loss_mse_ose[0] * self.norm_stats[phase][1] ** 2,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
            )
        
        self.log(
            f"{phase}_gloss_osse",
            loss_mse_osse[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_mse_osse",
            10000 * loss_mse_osse[0] * self.norm_stats[phase][1] ** 2,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
            )
        
        self.log(
            f"{phase}_loss",
            training_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        return training_loss, out_ose

class LitUnetSI(LitUnetOSEwOSSE):
    def __init__(self, config_x0, training_mode, w_end_to_end, w_si, n_steps_val, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_x0 = config_x0
        self.training_mode = training_mode
        self.w_end_to_end = w_end_to_end
        self.w_si = w_si
        self.n_steps_val = n_steps_val

    def sample_x0(self, batch, phase):
        if self.config_x0 == 'gaussian':
            return torch.randn(batch.input.size(),device=batch.input.device)
        elif self.config_x0 == 'gaussian+obs':
            return batch.input.nan_to_num() + 0.5 * torch.randn(batch.input.size(),device=batch.input.device)

    def base_step(self, batch, phase):

        if phase == 'train':
            return self.base_step_train(batch, phase)
        else:
            return self.base_step_end_to_end(batch, phase)

    def base_step_train_si(self, batch, phase):

        # sample x0
        x0 = self.sample_x0(batch, phase)

        # sample time values between 0 and 1
        # and associated xt states
        time_values = torch.rand((batch.input.size(0),1,1,1),device=batch.input.device).repeat(1,1,batch.input.size(2),batch.input.size(3))
        xt = (1-time_values.repeat(1,batch.input.size(1),1,1)) * x0 + time_values.repeat(1,batch.input.size(1),1,1) * batch.tgt.nan_to_num()

        # apply model
        batch_xt = TrainingItemOSEwOSSE(torch.cat((xt, batch.input.nan_to_num(),time_values), dim=1),
                                        None,None, None,
                                        batch.lon, batch.lat)
            
        return xt + self.solver(batch_xt)

    def base_step_end_to_end(self, batch, phase='val'):
        # sample x0
        x1_hat = self.sample_x0(batch, phase)

        #loop over a number of steps
        for k in range(self.n_steps_val):
            step = k / self.n_steps_val
            time_values = step * torch.ones((x1_hat.size(0),1,x1_hat.size(2),x1_hat.size(3)), device=x1_hat.device)

            batch_xt = TrainingItemOSEwOSSE(torch.cat((x1_hat, batch.input.nan_to_num(),time_values), dim=1),
                                            None,None, None,
                                            batch.lon, batch.lat)
            
            x1_hat = x1_hat + step * self.solver(batch_xt)

        x1_hat = x1_hat + self.solver(batch_xt)

        return x1_hat

    def forward(self, batch):
        """
        Forward pass through the solver.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: first output of the LatentSolver.
        """
        return self.base_step_end_to_end(batch) #self.solver(batch)[0]
    
    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:

            #print( batch.tgt.isfinite().float().mean() )
            #print( batch.input.isfinite().float().mean() )
            #print('\n ****')
            return None, None

        # SI training loss 
        if self.w_si > 0. :
            x1_hat = self.base_step_train_si(batch, phase)

            loss_mse = self.loss_mse(batch,x1_hat,phase)
            training_loss = self.w_si * self.w_ose * ( self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1] )

        else:
            training_loss = 0.

        # end-to-end training loss
        if self.w_end_to_end > 0. :
            x1_hat = self.base_step_end_to_end(batch, phase)
            
            loss_mse = self.loss_mse(batch,x1_hat,phase)
            training_loss =  training_loss + self.w_end_to_end * self.w_ose * ( self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1] )

        self.log(
            f"{phase}_gloss",
            loss_mse[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_mse",
            10000 * loss_mse[0] * self.norm_stats[phase][1] ** 2,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
            )
        
        self.log(
            f"{phase}_loss",
            training_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        return training_loss, x1_hat

