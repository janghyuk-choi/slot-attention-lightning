"""
source: https://github.com/karazijal/clevrtex/blob/master/experiments/genesisv2.py
"""

from typing import Any, List

import torch
import torch.nn.functional as F
import torchvision
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from utils.evaluator import ARIEvaluator, mIoUEvaluator
from utils.vis_utils import visualize


class LitGenesis2(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        name: str = "gen2",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # metric objects for calculating and averaging accuracy across batches
        self.train_fg_ari = ARIEvaluator()
        self.train_ari = ARIEvaluator()
        self.train_miou = mIoUEvaluator()

        self.val_mse = MeanMetric()
        self.val_fg_ari = ARIEvaluator()
        self.val_ari = ARIEvaluator()
        self.val_miou = mIoUEvaluator()

        self.val_fg_ari_best = MaxMetric()
        self.val_ari_best = MaxMetric()
        self.val_miou_best = MaxMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_elbo = MeanMetric()
        self.train_kl = MeanMetric()
        self.train_loss_comp_ratio = MeanMetric()
        self.train_recon_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        outputs = self.net(x)
        return outputs

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_fg_ari_best.reset()
        self.val_ari_best.reset()
        self.val_miou_best.reset()

    def model_step(self, batch: Any):
        img = batch["image"]
        outputs = self.net(img)
        loss = outputs["loss"]
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_elbo(outputs["elbo"])
        self.log("train/elbo", self.train_elbo, on_step=False, on_epoch=True, prog_bar=True)

        self.train_kl(outputs["kl"])
        self.log("train/kl", self.train_kl, on_step=False, on_epoch=True, prog_bar=True)

        self.train_loss_comp_ratio(outputs["rec_loss"] / outputs["kl"])
        self.log(
            "train/loss_comp_ratio",
            self.train_loss_comp_ratio,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_recon_loss(outputs["rec_loss"])
        self.log(
            "train/recon_loss", self.train_recon_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        _, outputs = self.model_step(batch)

        self.val_mse(
            F.mse_loss(outputs["canvas"], batch["image"], reduction="none").sum((1, 2, 3))
        )
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

        masks = batch["masks"].squeeze(-1)
        pred_masks = outputs["layers"]["mask"].squeeze(2)

        self.val_miou.evaluate(pred_masks, masks)
        self.val_ari.evaluate(pred_masks, masks)
        self.val_fg_ari.evaluate(pred_masks, masks[:, 1:])

        recons = torch.einsum("bkchw->bkhwc", outputs["layers"]["patch"])
        pred_masks = torch.einsum("bkchw->bkhwc", outputs["layers"]["mask"])
        B, K, H, W, _ = pred_masks.shape

        if batch_idx == 0:
            n_sampels = 4
            wandb_img_list = list()
            for vis_idx in range(n_sampels):
                vis = visualize(
                    image=batch["image"][vis_idx].unsqueeze(0),
                    recon_combined=outputs["canvas"][vis_idx].unsqueeze(0),
                    recons=recons[vis_idx].unsqueeze(0),
                    pred_masks=pred_masks[vis_idx].unsqueeze(0),
                    gt_masks=batch["masks"][vis_idx].unsqueeze(0),
                    attns=torch.zeros(
                        (1, 1, 1, H * W, K), dtype=torch.float32, device=pred_masks.device
                    ),  # dummy attns
                    colored_box=True,
                )
                grid = torchvision.utils.make_grid(vis, nrow=1, pad_value=0)
                wandb_img = wandb.Image(grid, caption=f"Epoch: {self.current_epoch}")
                wandb_img_list.append(wandb_img)
            self.logger.log_image(key="Visualization on Validation Set", images=wandb_img_list)

        return None

    def validation_epoch_end(self, outputs: List[Any]):
        val_fg_ari = self.val_fg_ari.get_results()
        self.val_fg_ari.reset()

        val_ari = self.val_ari.get_results()
        self.val_ari.reset()

        val_miou = self.val_miou.get_results()
        self.val_miou.reset()

        self.val_fg_ari_best(val_fg_ari)
        self.val_ari_best(val_ari)
        self.val_miou_best(val_miou)

        self.log_dict(
            {
                "val/fg-ari": val_fg_ari,
                "val/ari": val_ari,
                "val/miou": val_miou,
                "val/fg-ari_best": self.val_fg_ari_best.compute(),
                "val/ari_best": self.val_ari_best.compute(),
                "val/miou_best": self.val_miou_best.compute(),
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LitGenesis2(None, None, None)
