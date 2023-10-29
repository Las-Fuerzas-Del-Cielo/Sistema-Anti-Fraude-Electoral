from .geotr_core import *
import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from inv3d_util.misc import median_blur
from torch.optim.lr_scheduler import OneCycleLR
from einops import rearrange

warnings.filterwarnings('ignore')


class LitGeoTr(pl.LightningModule):
    dataset_options = {
        "resolution": 288,
    }

    train_options = {
        "max_epochs": 300,
        "batch_size": 8,
        "gradient_clip_val": 1,
        "early_stopping_patience": 25,
    }

    def __init__(self):
        super().__init__()
        self.model = GeoTr(num_attn_layers=6)
        self.epochs = None
        self.steps_per_epoch = None

    def forward(self, image, **kwargs):
        bm = self.model(image)
        bm = rearrange(bm, "b c h w -> b c w h") / 288
        bm = median_blur(bm)
        bm = torch.clamp(bm, min=0, max=1)
        return bm

    def training_step(self, batch, batch_idx):
        bm_true = rearrange(batch["train"]["bm"], "b c h w -> b c w h") * 288
        bm_pred = self.model(batch["input"]["image"])

        l1_loss = F.l1_loss(bm_pred, bm_true)
        mse_loss = F.mse_loss(bm_pred, bm_true)

        self.log("train/l1_288_loss", l1_loss)
        self.log("train/mse_288_loss", mse_loss)

        return l1_loss

    def validation_step(self, batch, batch_idx):
        bm_true = batch["train"]["bm"]

        bm_pred = self.model(batch["input"]["image"])
        bm_pred = rearrange(bm_pred, "b c h w -> b c w h") / 288

        self.log("val/mse_loss", F.mse_loss(bm_pred, bm_true), sync_dist=True)
        self.log("val/l1_loss", F.l1_loss(bm_pred, bm_true), sync_dist=True)

    def configure_optimizers(self):
        assert self.epochs is not None
        assert self.steps_per_epoch is not None

        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = OneCycleLR(optimizer, max_lr=10e-4, epochs=self.epochs,
                               steps_per_epoch=self.steps_per_epoch)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/mse_loss"
        }
