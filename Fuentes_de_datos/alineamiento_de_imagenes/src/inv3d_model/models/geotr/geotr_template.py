from .geotr_core import *
import warnings

import pytorch_lightning as pl
from inv3d_util.misc import median_blur
from torch.optim.lr_scheduler import OneCycleLR
from einops import rearrange

warnings.filterwarnings('ignore')


class LitGeoTrTemplate(pl.LightningModule):
    dataset_options = {
        "resolution": 288,
    }

    train_options = {
        "max_epochs": 300,
        "batch_size": 4,  # virtual batch size of 8
        "accumulate_grad_batches": 2,  # virtual batch size of 8
        "gradient_clip_val": 1,
        "early_stopping_patience": 25,
    }

    def __init__(self):
        super().__init__()
        self.model = GeoTrTemplate(num_attn_layers=6)
        self.epochs = None
        self.steps_per_epoch = None

    def forward(self, image, template, **kwargs):
        bm = self.model(image, template)
        bm = rearrange(bm, "b c h w -> b c w h") / 288
        bm = median_blur(bm)
        bm = torch.clamp(bm, min=0, max=1)
        return bm

    def training_step(self, batch, batch_idx):
        bm_true = rearrange(batch["train"]["bm"], "b c h w -> b c w h") * 288
        bm_pred = self.model(batch["input"]["image"], batch["input"]["template"])

        l1_loss = F.l1_loss(bm_pred, bm_true)
        mse_loss = F.mse_loss(bm_pred, bm_true)

        self.log("train/l1_288_loss", l1_loss)
        self.log("train/mse_288_loss", mse_loss)

        return l1_loss

    def validation_step(self, batch, batch_idx):
        bm_true = batch["train"]["bm"]

        bm_pred = self.model(batch["input"]["image"], batch["input"]["template"])
        bm_pred = rearrange(bm_pred, "b c h w -> b c w h") / 288

        self.log("val/mse_loss", F.mse_loss(bm_pred, bm_true), sync_dist=True)
        self.log("val/l1_loss", F.l1_loss(bm_pred, bm_true), sync_dist=True)

    def configure_optimizers(self):
        assert self.epochs is not None
        assert self.steps_per_epoch is not None

        optimizer = torch.optim.AdamW(self.parameters())
        scheduler = OneCycleLR(optimizer, max_lr=10e-4, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/mse_loss"
        }


class GeoTrTemplate(nn.Module):
    def __init__(self, num_attn_layers):
        super(GeoTrTemplate, self).__init__()
        self.num_attn_layers = num_attn_layers

        self.hidden_dim = hdim = 256

        self.fnet1 = BasicEncoder(output_dim=128, norm_fn='instance')
        self.fnet2 = BasicEncoder(output_dim=128, norm_fn='instance')

        self.TransEncoder = TransEncoder(self.num_attn_layers, hidden_dim=hdim)
        self.TransDecoder = TransDecoder(self.num_attn_layers, hidden_dim=hdim)
        self.query_embed = nn.Embedding(1296, self.hidden_dim)

        self.update_block = UpdateBlock(self.hidden_dim)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, template):
        fmap1 = self.fnet1(image1)  # torch.Size([N, 3, 288, 288]) -> torch.Size([4, 128, 36, 36])
        fmap1 = torch.relu(fmap1)

        fmap2 = self.fnet2(template)
        fmap2 = torch.relu(fmap2)

        fmap = torch.cat([fmap1, fmap2], dim=1)

        fmap = self.TransEncoder(fmap)
        fmap = self.TransDecoder(fmap, self.query_embed.weight)

        # convex upsample baesd on fmap
        coodslar, coords0, coords1 = self.initialize_flow(image1)
        coords1 = coords1.detach()
        mask, coords1 = self.update_block(fmap, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up

        return bm_up
