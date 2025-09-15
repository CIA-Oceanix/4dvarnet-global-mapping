from collections import namedtuple
import functools as ft
import time

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr
#from torchvision.transforms import v2

from ocean4dvarnet.data import BaseDataModule, TrainingItem
from ocean4dvarnet.models import Lit4dVarNet,GradSolver

TrainingItemwLonLat = namedtuple('TrainingItemwLonLat', ['input', 'tgt','lon','lat'])
TrainingItemOSEwOSSE = namedtuple('TrainingItemOSEwOSSE', ['input', 'tgt','input_osse','tgt_osse','lon','lat'])
TrainingItemOSEwOSSEwMask = namedtuple('TrainingItemOSEwOSSEwMask', ['input', 'tgt','input_osse','tgt_osse','lon','lat','mask_input_lr'])
PredictItem = namedtuple("PredictItem", ("input","lon","lat"))

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


class GradSolverZeroInit_withStep(GradSolver):


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

        t = torch.tensor([step], device=grad.device).repeat(grad.shape[0])
        gmod = self.grad_mod(grad, t)

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


### UNET SOLVER

# --- Bloc de base ResNet ---
class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, dropout=0.0, bias=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias)
        self.gn1 = torch.nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias)
        self.gn2 = torch.nn.GroupNorm(max(1, out_ch // 8), out_ch)

        self.act = lambda x: x * torch.sigmoid(x)  # Swish
        self.dense = Dense(embed_dim, out_ch)      # time embedding projection
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        # skip 1x1 conv si dimensions changent
        self.skip = torch.nn.Conv2d(in_ch, out_ch, 1, bias=bias) if in_ch != out_ch else torch.nn.Identity()

    def forward(self, x, embed):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h + self.dense(embed))
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h + self.dense(embed))

        return h + self.skip(x)


class GaussianFourierProjection(torch.nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    ret = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    return ret

class Dense(torch.nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = torch.nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]



class UnetGradModelUnet(torch.nn.Module):
    """Score-based UNet avec ResNet blocks et time embedding."""

    def __init__(self, dim_in, dim_hidden, embed_dim, num_levels, unet, out_activation=None, bias=False, dropout=0.0):
        super().__init__()

        # progression des canaux
        channels = [dim_hidden]
        for i in range(num_levels - 1):
            channels.append(channels[-1] * 2)

        # time embedding
        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            torch.nn.Linear(embed_dim, embed_dim)
        )
        self.act = lambda x: x * torch.sigmoid(x)
        self.norm = torch.nn.Parameter(torch.tensor([1.]))

        # --- Encoding ---
        self.enc_blocks = torch.nn.ModuleList()
        in_ch = dim_in
        for ch in channels:
            self.enc_blocks.append(ResBlock(in_ch, ch, embed_dim, bias=bias, dropout=dropout))
            in_ch = ch

        # --- Bottleneck ---
        self.bottleneck = ResBlock(channels[-1], channels[-1], embed_dim, dropout=dropout)

        # --- Decoding ---
        self.dec_blocks = torch.nn.ModuleList()
        for i in reversed(range(1, len(channels))):
            self.dec_blocks.append(
                torch.nn.ModuleDict({
                    "upsample": torch.nn.ConvTranspose2d(channels[i], channels[i-1], 4, stride=2, padding=1, bias=bias),
                    "resblock": ResBlock(channels[i-1]*2, channels[i-1], embed_dim, dropout=dropout, bias=bias)
                })
            )

        # --- Final ---
        if unet is not None:
            self.conv_out = unet
            self.use_unet = True
        else:
            self.use_unet = False
            self.conv_out = torch.nn.Conv2d(channels[0]*2, dim_in, 3, padding=1)


        # Option activation de sortie
        if out_activation == "tanh":
            self.out_act = torch.nn.Tanh()
        elif out_activation == "sigmoid":
            self.out_act = torch.nn.Sigmoid()
        else:
            self.out_act = torch.nn.Identity()

    def reset_state(self, inp):
        self._grad_norm = None

    def forward(self, x, t):
        if self._grad_norm is None:
            self._grad_norm = (x ** 2).mean().sqrt()
        x = x / self._grad_norm

        # time embedding 
        embed = self.act(self.embed(t))

        # --- Encoder ---
        hs = []
        h = x
        for block in self.enc_blocks:
            h = block(h, embed)
            hs.append(h)
            h = torch.nn.functional.avg_pool2d(h, 2) if block != self.enc_blocks[-1] else h  # downsample sauf dernier

        # --- Bottleneck ---
        h = self.bottleneck(h, embed)

        # --- Decoder ---
        skip_connections = hs[::-1]
        for skip, dec in zip(skip_connections[1:], self.dec_blocks):  # on garde aussi skip du niveau le + bas
            h = dec["upsample"](h)
            h = dec["resblock"](torch.cat([h, skip], dim=1), embed)

        # --- Final ---
        if self.use_unet == True:
            out = self.conv_out.predict(torch.cat([h, skip_connections[-1]], dim=1))
        else:
            out = self.conv_out(torch.cat([h, skip_connections[-1]], dim=1))

        return self.out_act(out)

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

        self.osse_with_interp_error = kwargs.pop("osse_with_interp_error",False)

        print( self.osse_with_interp_error)

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

    def sample_osse_data_with_l3interp_errr(self,batch):
        # to be implemented in child class if needed
        # patch dimensions
        K = batch.input.shape[0]
        N = batch.input.shape[2]
        M = batch.input.shape[3]
        T = batch.input.shape[1]

        # start with time interpolation error only
        dt =  1. * ( torch.rand((K,T-2,N,M), dtype=torch.float32, device=batch.input.device) - 0.5 )
        dt = torch.nn.functional.avg_pool2d(dt,(4,4))
        dt = torch.nn.functional.interpolate(dt,scale_factor=4.,mode='bilinear')
          
        inp_dt = (dt >= 0. ) * ( (1.-dt) * batch.tgt[:,1:-1,:,:] + dt * batch.tgt[:,2:,:,:] )
        inp_dt += ( dt < 0.) * ( (1+dt) * batch.tgt[:,1:-1,:,:] - dt * batch.tgt[:,0:-2,:,:] )
        inp_dt = torch.cat( ( batch.tgt[:,0:1,:,:], inp_dt, batch.tgt[:,-1:,:,:] ), dim = 1)

        #print('\n ....... interpolation time' )
        #print(batch.tgt[0,0:5,10,10].detach().cpu().numpy(),inp_dt[0,1,10,10].detach().cpu().numpy() , dt[0,0,10,10].detach().cpu().numpy()     )

        # space interpolation error
        scale_spatial_perturbation = 1.
        dx = scale_spatial_perturbation * torch.rand((K,T,N,M), dtype=torch.float32, device=batch.input.device)
        dy = scale_spatial_perturbation * torch.rand((K,T,N,M), dtype=torch.float32, device=batch.input.device)

        dx = torch.nn.functional.avg_pool2d(dx,(4,4))
        dx = torch.nn.functional.interpolate(dx,scale_factor=4.,mode='bilinear')

        dy = torch.nn.functional.avg_pool2d(dy,(4,4))
        dy = torch.nn.functional.interpolate(dy,scale_factor=4.,mode='bilinear')

        dx = dx[:,:,:-1,:-1]
        dy = dy[:,:,:-1,:-1]

        inp_dxdydt = inp_dt[:,:,:-1,:-1] * (1-dx) * (1-dy) 
        inp_dxdydt += inp_dt[:,:,:-1,1:] * (1-dx) * dy
        inp_dxdydt += inp_dt[:,:,1:,:-1] * dx * (1-dy)
        inp_dxdydt += inp_dt[:,:,1:,1:] * dx * dy

        inp_dxdydt = torch.where( inp_dxdydt.isfinite() , inp_dxdydt, batch.tgt[:,:,:-1,:-1] )

        #print('\n....... interpolation space' )
        #print(inp_dt[0,2,10,10].detach().cpu().numpy(),inp_dxdydt[0,2,10,10].detach().cpu().numpy(), dx[0,2,10,10].detach().cpu().numpy(), dy[0,2,10,10].detach().cpu().numpy() )


        inp_dxdydt = torch.cat( ( inp_dxdydt, inp_dt[:,:,-1:,:-1] ), dim = 2)
        inp_dxdydt = torch.cat( ( inp_dxdydt, inp_dt[:,:,:,-1:] ), dim = 3)

        input = torch.where( batch.input.isfinite() ,  batch.input + inp_dxdydt.detach() - batch.tgt , torch.nan )
        #input = torch.where( batch.input.isfinite() ,  inp_dxdydt.detach() , torch.nan )
        input = input.detach()

        display = None #True #
        if display is not None:
            noise = ( input - batch.tgt ) * self.norm_stats['train'][1]
            print("..... mean, std of the simulated spatial perturnation noise : %.3f -- %.3f "%(torch.nanmean(noise), torch.sqrt( torch.nanmean(noise**2) - torch.nanmean(noise)**2) ))
            #print("..... number of observed pixels (new) : %.2f  "%(100 * input.isfinite().float().mean()) )
            #print("..... number of observed pixels (new) : %.2f  "%(100 * noise.isfinite().float().mean()) )
            #print("..... number of observed pixels (orig): %.2f "%(100 * batch.input.isfinite().float().mean()) )


        return TrainingItem(input, batch.tgt)


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

        # osse input
        if ( self.osse_with_interp_error == True ) and ( ( phase == "train" ) or ( phase == "val" ) ):
        
            batch_ = self.sample_osse_data_with_l3interp_errr(batch)
        else:
            batch_ = batch
            
        # apply base-step
        loss, out = self.base_step(batch_, phase)

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

def cosanneal_lr_adam_unet_with_preposprocessing(lit_mod, lr, T_max=100, weight_decay=0., freeze_pretrained_model=False):
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
    if freeze_pretrained_model:
        for param in lit_mod.solver.parameters():
            param.requires_grad = False
        lr_solver = 0.
    else:
        lr_solver = lr

    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.parameters(), "lr": lr_solver},
            {"params": lit_mod.pre_pro_model.parameters(), "lr": lr},
            {"params": lit_mod.post_pro_model.parameters(), "lr": lr},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max
        ),
    }

def cosanneal_lr_adam_fdv_with_preposprocessing(lit_mod, lr, T_max=100, weight_decay=0., freeze_pretrained_model=False):
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
    if freeze_pretrained_model:
        for param in lit_mod.solver.parameters():
            param.requires_grad = False
        lr_solver = 0.
    else:
        lr_solver = lr

    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
            {"params": lit_mod.pre_pro_model.parameters(), "lr": lr},
            {"params": lit_mod.post_pro_model.parameters(), "lr": lr},
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

def cosanneal_lr_adam_model_with_preposprocessing(lit_mod, lr, T_max=100, weight_decay=0., freeze_pretrained_model=False):
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
    if freeze_pretrained_model:
        for param in lit_mod.solver.parameters():
            param.requires_grad = False
        lr_solver = 0.
    else:
        lr_solver = lr

    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr_solver},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr_solver},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr_solver / 2},
            {"params": lit_mod.pre_pro_model.parameters(), "lr": lr},
            {"params": lit_mod.post_pro_model.parameters(), "lr": lr},
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

        # osse input
        if ( self.osse_with_interp_error == True ) and ( ( phase == "train" ) or ( phase == "val" ) ) :
            batch_ = self.sample_osse_data_with_l3interp_errr(batch)
        else:
            batch_ = batch

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
                 w_ose_obs=0,
                 frac_random_gaps=0.9,
                 width_lat_gaps=32, 
                 width_lon_gaps=4,
                 width_time_gaps=2,
                 idx_sensor_for_val_metrics=11,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.w_ose = w_ose
        self.w_osse = w_osse
        self.scale_loss_ose = scale_loss_ose
        self.osse_type = osse_type
        self.sig_noise_ose2osse = sig_noise_ose2osse

        self.patch_normalization = patch_normalization
        self.normalization_noise = normalization_noise
        self.frac_random_gaps = frac_random_gaps
        self.w_ose_obs = w_ose_obs
        self.idx_sensor_for_val_metrics = idx_sensor_for_val_metrics

        self.width_lat_gaps = width_lat_gaps
        self.width_lon_gaps = width_lon_gaps
        self.width_time_gaps = width_time_gaps
        if self.frac_random_gaps <= 0.0 :
            self.remove_random_gaps = False
            self.w_ose_obs = 0.
        else:   
            self.remove_random_gaps = True


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

        elif osse_type == 'osse-with-l3interp-error':

            batch_osse = TrainingItemOSEwOSSE(batch.input_osse,
                                          batch.tgt_osse,
                                          None, None,
                                          batch.lon,
                                          batch.lat)

            batch_osse = self.sample_osse_data_with_l3interp_errr(batch_osse)

            return TrainingItemOSEwOSSE(batch.input,
                                        batch.tgt,
                                        batch_osse.input,
                                        batch_osse.tgt,
                                        batch.lon,
                                        batch.lat)
        return batch

    def mask_random_gaps_in_batch(self,batch,frac_missing=0.001):

        
        # randomly remove some observations in the input data
        frac_missing = ( 1 + 0.25 * ( torch.rand( (batch.input.shape[0] ,1,1,1) , device=batch.input.device ) - 0.5 ) ) * frac_missing
        frac_missing = frac_missing.view(-1,1,1,1).repeat(1,self.width_time_gaps,batch.input.shape[2],batch.input.shape[3])
        mask = ( torch.rand( (batch.input.shape[0],self.width_time_gaps,batch.input.shape[2],batch.input.shape[3]) , device=batch.input.device ) > 1. - frac_missing ).float()
        dt_ = int( (batch.input.shape[1]-self.width_time_gaps)/2 )
        mask_zeros = torch.zeros( (batch.input.shape[0],dt_,batch.input.shape[2],batch.input.shape[3]) , device=batch.input.device )
        mask = torch.cat( (mask_zeros, mask, mask_zeros), dim=1)

        mask = torch.nn.functional.avg_pool2d(mask,(self.width_lat_gaps,self.width_lon_gaps))
        mask = (mask > 0.0).float()
        mask = torch.nn.functional.interpolate(mask, scale_factor=(self.width_lat_gaps,self.width_lon_gaps), mode='bilinear')
        mask = (mask > 0.0).float()

        # random rotation (problem with installation of torchvision)
        #rotater = v2.RandomRotation(degrees=(-10, 10))
        #mask = rotater(mask)

        input_ose = torch.where( mask == 0., batch.input , float('nan'))
        tgt_ose = torch.where( mask == 1., batch.input , float('nan'))

        display = None #True #True
        if display is not None :
            print(" ....Percentage of obs pixels", 100.*batch.input.isfinite().float().mean().detach().cpu().numpy(), flush=True)
            print(" ....Percentage of kept obs pixels", 100.*input_ose.isfinite().float().mean().detach().cpu().numpy(), flush=True)
            print(" ....Percentage of removed obs pixels", 100.*tgt_ose.isfinite().float().mean().detach().cpu().numpy(), flush=True)
            print(" ....Intersection of kept and removed obs pixels", 100.*(input_ose.isfinite().float() * tgt_ose.isfinite().float()).mean().detach().cpu().numpy(), flush=True)

        return TrainingItemOSEwOSSE(input_ose,
                                        tgt_ose,
                                        None,None,
                                        batch.lon,
                                        batch.lat)


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

        if phase == 'test' :
            return PredictItem((batch.input- m_new) / s_new , 
                                None,
                                batch.lon,
                                batch.lat), m_new, s_new

            #TrainingItemOSEwOSSE((batch.input- m_new) / s_new ,
            #                              None,
            #                              None, None,
            #                              batch.lon,
            #                              batch.lat), m_new, s_new
        else:
            return TrainingItemOSEwOSSE((batch.input- m_new) / s_new ,
                                          (batch.tgt - m_new) / s_new,
                                          None, None,
                                          batch.lon,
                                          batch.lat), m_new, s_new


    def forward(self, batch):
        """
        Forward pass through the solver.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: first output of the LatentSolver.
        """

        return self.forward_ose(batch,phase='test') #self.solver(batch)[0]
    
    def forward_ose(self, batch,phase):
        """
        Forward pass through the solver for OSE data.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: first output of the LatentSolver.
        """

        # patch-based normalisation for OSE patches
        if self.patch_normalization is not None :   
            batch_,m_new, s_new = self.apply_patch_normalization(batch,phase)
        else:   
            batch_ = batch
            m_new, s_new = 0., 1.

        # apply solver to ose patches
        out_ose = self.solver(batch_)
        out_ose = (out_ose * s_new) + m_new

        return out_ose

    def remove_randomly_one_sensor(self,batch,phase='train'):
        display= None #True # True

        # randomly remove available altimetters
        avail_nadir = torch.sum( batch.mask_input_lr , dim=(1,2,3) ) > 1.
        if phase == 'train':
            rand_vals = torch.rand( avail_nadir.shape , device=batch.input.device ) * avail_nadir.float()
        elif phase == 'val':
            rand_vals = torch.zeros_like( avail_nadir , device=batch.input.device ).float()
            rand_vals[:,self.idx_sensor_for_val_metrics] = 1.
            rand_vals = rand_vals.detach()

        max_rand = torch.max( rand_vals , dim = 1 , keepdim=True).values.repeat(1,avail_nadir.shape[1])

        if False : #True : # 
            mask = (rand_vals < max_rand).float() * avail_nadir.float()
            mask = batch.mask_input_lr.float() * mask.view(-1,1,1,1,batch.mask_input_lr.shape[-1]).repeat(1,batch.mask_input_lr.shape[1],batch.mask_input_lr.shape[2],batch.mask_input_lr.shape[3],1)
            mask = torch.sum( mask , dim = -1 ) > 0.9
            mask = torch.nn.functional.interpolate(mask.float(),scale_factor=4.,mode='bilinear') >= 0.5
            mask = 1. - mask.float()
        else:
            mask = (rand_vals == max_rand).float() * avail_nadir.float()
            mask = batch.mask_input_lr.float() * mask.view(-1,1,1,1,batch.mask_input_lr.shape[-1]).repeat(1,batch.mask_input_lr.shape[1],batch.mask_input_lr.shape[2],batch.mask_input_lr.shape[3],1)
            mask = torch.sum( mask , dim = -1 ) > 0.9
            mask = torch.nn.functional.interpolate(mask.float(),scale_factor=4.,mode='bilinear') >= 0.5
            mask = mask.float()

        if display is not None :
            print('\n...... Selected nadir : ',torch.argmax(rand_vals,dim=1), flush=True)

            print(" ....Percentage of obs lr pixels", 100.*batch.mask_input_lr.float().mean().detach().cpu().numpy(), flush=True)
            #print(" ....Percentage of kept obs lr pixels", 100.*mask.float().mean().detach().cpu().numpy(), flush=True)


        input_ose = torch.where( mask == 0., batch.input , float('nan'))
        tgt_ose = torch.where( mask == 1., batch.input , float('nan'))

        if display is not None :
            print(" ....Percentage of obs pixels", 100.*batch.input.isfinite().float().mean().detach().cpu().numpy(), flush=True)
            print(" ....Percentage of kept obs pixels", 100.*input_ose.isfinite().float().mean().detach().cpu().numpy(), flush=True)
            print(" ....Percentage of removed obs pixels", 100.*tgt_ose.isfinite().float().mean().detach().cpu().numpy(), flush=True)
            print(" ....Intersection of kept and removed obs pixels", 100.*(input_ose.isfinite().float() * tgt_ose.isfinite().float()).mean().detach().cpu().numpy(), flush=True)

        return TrainingItemOSEwOSSE(input_ose,
                                        tgt_ose,
                                        None,None,
                                        batch.lon,
                                        batch.lat)
    def base_step(self, batch, phase):

        display = None # True #None

        # remove obs data
        if self.remove_random_gaps :
            # random gaps (rectangular boxes    )
            #batch_ = self.mask_random_gaps_in_batch(batch,frac_missing=self.frac_random_gaps)

            # randomly remove available altimetters
            batch_ = self.remove_randomly_one_sensor(batch,phase)
        else:
            batch_  = batch

        # apply solver to ose patches
        out_ose = self.forward_ose(batch_,phase)

        return out_ose, batch_ #ose


    def base_step_osse(self, batch, phase):
        # sampling OSSE input data
        if self.w_osse > 0. :
            batch_ = self.aug_data_with_ose2osse_noise(batch,
                                                       sig_noise_ose2osse=self.sig_noise_ose2osse,
                                                       osse_type=self.osse_type)

            batch_osse = TrainingItemOSEwOSSE(batch_.input_osse,
                                            batch_.tgt_osse,
                                            None, None,
                                            batch.lon,
                                            batch.lat)

            # apply solver to ose patches
            out_osse = self.solver(batch_osse)
        else:
            out_osse = None
            batch_osse = None

        return out_osse, batch_osse


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

        # OSE data
        out_ose, batch_ose = self.base_step(batch, phase)

        loss_mse_ose_lr = self.loss_mse_lr(batch,out_ose,phase,scale=self.scale_loss_ose)
        loss_mse_ose_hr = self.loss_mse(batch_ose,out_ose,phase)
        loss_mse_ose_hr = loss_mse_ose_hr[0], 0.

        #print('.... mse : ', np.sqrt( loss_mse_ose_hr[0].detach().cpu().numpy()))

        # OSSE data
        if self.w_osse > 0.:
            out_osse, batch_osse = self.base_step_osse(batch, phase)
            loss_mse_osse = self.loss_mse(batch_osse,out_osse,phase)
        else:
            out_osse, batch_osse = None, None
            loss_mse_osse = 0., 0.

        training_loss  = self.w_ose * ( self.w_mse_lr * loss_mse_ose_lr[0] + self.w_grad_mse_lr * loss_mse_ose_lr[1] )
        training_loss += self.w_ose_obs * ( self.w_mse * loss_mse_ose_hr[0] + self.w_grad_mse * loss_mse_ose_hr[1] )
        training_loss += self.w_osse * ( self.w_mse * loss_mse_osse[0] + self.w_grad_mse * loss_mse_osse[1] )

        #print("loss lr", loss_mse_ose_lr[0].detach().cpu().numpy(),
        #      "gloss lr", loss_mse_ose_lr[1].detach().cpu().numpy(),
        #      " loss hr", loss_mse_ose_hr[0].detach().cpu().numpy(),
        #      " loss osse", loss_mse_osse[0].detach().cpu().numpy(), flush=True)

        self.log(
            f"{phase}_gloss_lr",
            loss_mse_ose_lr[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )
        self.log(
            f"{phase}_mse",
            10000 * loss_mse_ose_hr[0] * self.norm_stats[phase][1] ** 2,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
            )
        
        self.log(
            f"{phase}_mse_lr",
            10000 * loss_mse_ose_lr[0] * self.norm_stats[phase][1] ** 2,
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
            10000 * loss_mse_osse[0] * self.norm_stats[phase][3] ** 2,
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


class PreProcessingModel(torch.nn.Module):
    def __init__(self, model=None,use_lonlat_in_preprocessing=True):
        super().__init__()

        if model is None:
            self.model = None
            self.use_lonlat_in_preprocessing = False
        else:
            self.model = model
            self.use_lonlat_in_preprocessing = use_lonlat_in_preprocessing

    def forward(self, x , z=None):
        if self.model is not None :
            if self.use_lonlat_in_preprocessing == True:
                x_ = torch.cat( (x,z), dim=1)
            else:
                x_ = x

            y = x + self.model.predict(x_.nan_to_num())

            return torch.where( x.isfinite(), y, float('nan'))
        else:
            return x

class PostProcessingModel(torch.nn.Module):
    def __init__(self, model=None,use_lonlat_in_preprocessing=True):
        super().__init__()

        if model is None:
            self.model = None
            self.use_lonlat_in_preprocessing = False
        else:
            self.model = model
            self.use_lonlat_in_preprocessing = use_lonlat_in_preprocessing

    def forward(self, x , z=None):

        if self.model is not None :
            if self.use_lonlat_in_preprocessing == True:
                x_ = torch.cat( (x,z), dim=1)
            else:
                x_ = x

            return self.model.predict(x_)
        else:
            return x


    
class LitUnetOSEwOSSEwithPrePostProcessing(LitUnetOSEwOSSE):
    def __init__(self, 
                 pre_pro_model, post_pro_model,
                 w_ose, w_osse, scale_loss_ose, osse_type, sig_noise_ose2osse, 
                 patch_normalization=None, normalization_noise=0.,
                 w_ose_obs=0,
                 frac_random_gaps=0.9,
                 width_lat_gaps=32, 
                 width_lon_gaps=4,
                 width_time_gaps=2,
                 *args, **kwargs):
        super().__init__(w_ose, w_osse, scale_loss_ose, osse_type, sig_noise_ose2osse,
                         patch_normalization, normalization_noise, w_ose_obs,
                         frac_random_gaps, width_lat_gaps, width_lon_gaps, width_time_gaps, *args, **kwargs)

        self.pre_pro_model = pre_pro_model
        self.post_pro_model = post_pro_model

    def forward_ose(self, batch,phase):
        """
        Forward pass through the solver for OSE data.

        Args:
            batch (dict): Input batch.

        Returns:
            torch.Tensor: first output of the LatentSolver.
        """

        # pre-process lon,lat
        if self.pre_pro_model.use_lonlat_in_preprocessing == True :
            lat = _LAT_TO_RAD * batch.lat.view(-1,1,batch.lat.shape[1],1).repeat(1,1,1,batch.input.shape[-1])
            lon = _LAT_TO_RAD * batch.lon.view(-1,1,1,batch.lon.shape[1]).repeat(1,1,batch.input.shape[2],1)

            z = torch.cat( (torch.cos(lat),torch.cos(lon),torch.sin(lon)),dim=1)
        else:
            z = None

        batch_ = TrainingItemOSEwOSSE(self.pre_pro_model(batch.input,z),
                                      None,
                                      None, None,
                                      batch.lon,
                                      batch.lat)

        # interpolation solver
        out = super().forward_ose(batch_,phase)

        # post-process model
        if self.post_pro_model.use_lonlat_in_preprocessing == True :
            lat = _LAT_TO_RAD * batch.lat.view(-1,1,batch.lat.shape[1],1).repeat(1,1,1,batch.input.shape[-1])
            lon = _LAT_TO_RAD * batch.lon.view(-1,1,1,batch.lon.shape[1]).repeat(1,1,batch.input.shape[2],1)

            z = torch.cat( (torch.cos(lat),torch.cos(lon),torch.sin(lon)),dim=1)
        else:
            z = None

        out_ose = self.post_pro_model(out,z)
 
        return out_ose

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

