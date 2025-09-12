import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import sys
import os
import random
import pickle
from itertools import chain

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import TrainingItem

from src.models import GradSolver
from src.data import BaseDataModule
from src.models import Lit4dVarNet


class Lit4dVarNet_option2(Lit4dVarNet):
    def test_step(self, batch, batch_idx, n_passes: int = 20):
        if batch_idx == 0:
            self.test_data = []

        m, s = self.norm_stats
        preds = []

        for _ in range(n_passes):
            out = self(batch=batch)
            preds.append(out.squeeze(dim=-1).detach().cpu())

        # Moyenne des prédictions
        mean_out = torch.stack(preds, dim=0).mean(dim=0)

        # Stockage des entrées, cibles, et prédictions moyennes
        self.test_data.append(torch.stack(
            [
                batch.input.cpu() * s + m,
                batch.tgt.cpu() * s + m,
                mean_out * s + m,
            ],
            dim=1,
        ))

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            # Sauvegarde des données test
            #self.logger.log_metrics(mean_metrics.add_suffix('_mean').to_dict())
            #self.logger.log_metrics(var_metrics.add_suffix('_var').to_dict())
            metrics.to_csv(Path(self.logger.log_dir) / 'metrics.csv')





class Lit4dVarNet_option1(Lit4dVarNet):
    def test_step(self, batch, batch_idx, n_passes: int = 20):
        if batch_idx == 0:
            self.test_data = []

        m, s = self.norm_stats
        preds = []

        for _ in range(n_passes):
            out = self(batch=batch)
            preds.append(torch.stack(
            [
                batch.input.cpu() * s + m,
                batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
            ))
        self.test_data.append(preds)

    def on_test_epoch_end(self):
        self.test_data = list(map(list, zip(*self.test_data)))
        all_metrics = []

        for i in range(len(self.test_data)):

            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data[i], self.rec_weight.cpu().numpy()
            )

            if isinstance(rec_da, list):
                rec_da = rec_da[0]

            test_data = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')

            metric_data = test_data.pipe(self.pre_metric_fn)
            metrics = {
                metric_n: metric_fn(metric_data) 
                for metric_n, metric_fn in self.metrics.items()
            }

            all_metrics.append(metrics)

            #print(metrics.to_frame(name="Metrics").to_markdown())

        df_metrics = pd.DataFrame(all_metrics)
        mean_metrics = df_metrics.mean()
        var_metrics = df_metrics.std()

        # Affichage / log
        print("\n--- Test Metrics ---")
        print("Moyennes :")
        print(mean_metrics.to_frame(name="Mean").to_markdown())
        print("\nVariances :")
        print(var_metrics.to_frame(name="Variance").to_markdown())

        if self.logger:
            # Sauvegarde des données test
            #self.logger.log_metrics(mean_metrics.add_suffix('_mean').to_dict())
            #self.logger.log_metrics(var_metrics.add_suffix('_var').to_dict())
            mean_metrics.add_suffix('_mean').to_csv(Path(self.logger.log_dir) / 'metrics_mean.csv')
            var_metrics.add_suffix('_var').to_csv(Path(self.logger.log_dir) / 'metrics_var.csv')






class BaseDataModule(BaseDataModule):
        def predict_dataloader(self):
            return self.test_dataloader()


def load_qg_data(path, obs_from_tgt=False):
    ds = xr.open_dataset(path).load()

    # Crée les champs 'input' et 'tgt'
    ds = ds.assign(
        input=ds['sf_obs'],
        tgt=ds['stream_function']
    )

    # Vérifie que les champs demandés existent
    for field in TrainingItem._fields:
        if field not in ds:
            raise ValueError(f"Champ manquant dans le dataset: {field}")

    # Extraire les champs dans le bon ordre
    ds_selected = ds[list(TrainingItem._fields)]

    # Transposer et convertir en array
    return ds_selected.transpose("time", "lat", "lon").to_array()



def base_inference(trainer: pl.Trainer, lit_mod, dm, ckpt: str):
    trainer.test(lit_mod, datamodule=dm, ckpt_path=ckpt)




class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True):
        super().__init__()
        self.bilin_quad = bilin_quad
        self.conv_in = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,padding_mode='circular'
        )
        self.conv_hidden = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,padding_mode='circular'
        )

        self.bilin_1 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,padding_mode='circular'
        )
        self.bilin_21 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,padding_mode='circular'
        )
        self.bilin_22 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,padding_mode='circular'
        )

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2,padding_mode='circular'
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward_ae(self, x):
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state))








class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,padding_mode='circular'
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2,padding_mode='circular'
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm
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



class GradSolver(GradSolver):
    def __init__(self, *args, **kwargs):
        self.init_type = kwargs.pop('init_type', None)
        self.max_sig = kwargs.pop('max_sig', None)
        self.test_sig = kwargs.pop('test_sig', None)
        super().__init__(*args, **kwargs)

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init
        elif self.init_type == 1:
            return batch.input.nan_to_num().detach().requires_grad_(True)
        elif self.init_type == 2:
            return torch.zeros_like(batch.input).detach().requires_grad_(True)
        elif self.init_type == 3:
            if self.training:
                sigma = random.uniform(0, self.max_sig)
            else:
                sigma = self.test_sig
            return (sigma * torch.randn_like(batch.input)).detach().requires_grad_(True)

