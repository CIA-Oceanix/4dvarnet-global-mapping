class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def __init__(self,  
                 w_mse,w_grad_mse, w_prior,
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
