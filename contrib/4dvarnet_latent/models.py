import functools as ft
import time

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr

from ocean4dvarnet.data import BaseDataModule, TrainingItem
from ocean4dvarnet.models import Lit4dVarNet,GradSolver

_LAT_TO_RAD = np.pi / 180.0

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


    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out)) # Why using init_state ?
        self.log(
            f"{phase}_gloss",
            grad_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost


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

        # log
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
        
        training_loss = self.w_mse * loss_mse_hr[0] + self.w_grad_mse * loss_mse_hr[1] 
        training_loss += self.w_mse_lr * loss_mse_lr[0] + self.w_grad_mse_lr * loss_mse_lr[1]
        training_loss += self.w_prior * loss_prior[0] + self.w_prior * loss_prior[1]
        training_loss += self.w_latent_ae * loss_latent_ae 

        return training_loss, out

class UnetSolver(torch.nn.Module):
    def __init__(self, dim_in, channel_dims, max_depth=None):
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
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
            ),
            torch.nn.ReLU(),
        )

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=dim_in,
                padding="same",
                kernel_size=3,
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
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
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
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
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
                x = funct.pad(x, (0, pad_w, 0, pad_h), mode="reflect", value=0)

            return torch.concat((x, residue), dim=1)
        else:
            return x


class UnetSolver2(UnetSolver):
    def __init__(self, dim_in, channel_dims, max_depth=None,dim_out=None):
        super().__init__(dim_in, channel_dims, max_depth)

        if dim_out is None :
            dim_out = dim_in

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4*dim_out,
                padding="same",
                kernel_size=3,
            ) )
        self.final_linear = torch.nn.Sequential(torch.nn.Linear(4*dim_out, dim_out))

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