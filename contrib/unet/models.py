import torch
import pytorch_lightning as pl
import torch.nn.functional as F

import pandas as pd
from pathlib import Path


class Unet(pl.LightningModule):
    def __init__(
        self,
        solver,
        channel_dims,
        rec_weight,
        opt_fn,
        norm_stats=None,
        test_metrics=None,
        pre_metric_fn=None,
        persist_rw=True,
        batch_selector=None,
    ):
        super().__init__()
        self.register_buffer(
            "rec_weight", torch.from_numpy(rec_weight), persistent=persist_rw
        )
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)

        self.max_depth = len(channel_dims) // 3
        self.solver = solver(channel_dims=channel_dims, max_depth=self.max_depth)

    # PYTORCH LIGHTNING LOGIC

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0.0, 1.0)

    @staticmethod
    def weighted_mse(err, weight):
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def forward(self, batch):
        return self.solver(batch)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None

        out = self(batch=batch.input)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)
        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    @property
    def test_quantities(self):
        return ["out"]

    def clear_gpu_mem(self):
        del self.solver

        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch.input)
        m, s = self.norm_stats

        self.test_data.append(
            torch.stack(
                [
                    out.squeeze(dim=-1).detach().cpu() * s + m,
                ],
                dim=1,
            )
        )

    def get_dT(self):
        return self.rec_weight.size()[0]

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
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())


class UnetSolver(torch.nn.Module):
    def __init__(self, dim_in, channel_dims, max_depth):
        super().__init__()
        self.max_depth = max_depth

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

    def forward(self, x):
        x = x.nan_to_num()
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

    # def concat_residue(self, x):
    #    if len(self.residues) != 0:
    #        return torch.concat((x, self.residues.pop(-1)), dim=1)
    #    else:
    #        return x

    def concat_residue(self, x):
        if len(self.residues) != 0:
            residue = self.residues.pop(-1)

            _, _, h_x, w_x = x.shape
            _, _, h_r, w_r = residue.shape

            pad_h = h_r - h_x
            pad_w = w_r - w_x

            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect", value=0)

            return torch.concat((x, residue), dim=1)
        else:
            return x


class UnetSolver3D(UnetSolver):
    def __init__(self, dim_in, channel_dims, max_depth):
        self.max_depth = max_depth

        self.ups = torch.nn.ModuleList()
        self.up_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.residues = list()

        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv3d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
            ),
            torch.nn.ReLU(),
        )

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv3d(
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
                    torch.nn.Conv3d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv3d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.up_pools.append(
                torch.nn.ConvTranspose3d(
                    in_channels=channel_dims[depth * 3 + 3],
                    out_channels=channel_dims[depth * 3 + 2],
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv3d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv3d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.down_pools.append(torch.nn.MaxPool3d(kernel_size=(1, 2, 2)))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = x.nan_to_num()
        x = self.final_up(self.unet_step(x, depth=0))
        x = x.squeeze(dim=1)
        x = torch.permute(x, dims=(0, 2, 3, 1))
        x = self.final_linear(x)
        x = torch.permute(x, dims=(0, 3, 1, 2))
        return x

    def concat_residue(self, x):
        if len(self.residues) != 0:
            residue = self.residues.pop(-1)

            _, _, _, h_x, w_x = x.shape
            _, _, _, h_r, w_r = residue.shape

            pad_h = h_r - h_x
            pad_w = w_r - w_x

            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect", value=0)

            return torch.concat((x, residue), dim=1)
        else:
            return x
