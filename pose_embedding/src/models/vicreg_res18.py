# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
# The projection head is the same as the Barlow Twins one
import csv
from datetime import datetime

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from lightly.models.modules import BarlowTwinsProjectionHead
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from lightly.loss import VICRegLoss


class Normalize(torch.nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)  # L2 normalization

class VICReg(pl.LightningModule):
    def __init__(
        self,
        optimizer,
        pretrained: bool = False,
        head_input_dim: int = 16,
        head_hidden_dim: int = 16,
        head_out_dim: int = 8,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        batch_size: int = 512,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.head_input_dim = head_input_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_out_dim = head_out_dim
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.input_dropout = input_dropout

        self.was_vis = False

        self.test_embeds = []

        if self.pretrained:
            self.weights = ResNet18_Weights.DEFAULT
        else:
            self.weights = None

        resnet = resnet18(weights=self.weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.backbone.append(nn.Flatten())
        self.backbone.append(nn.Linear(512, 256))
        self.backbone.append(nn.BatchNorm1d(256))
        self.backbone.append(nn.LeakyReLU(inplace=True))

        self.backbone.append(nn.Linear(256, 128))
        self.backbone.append(nn.BatchNorm1d(128))
        self.backbone.append(nn.LeakyReLU(inplace=True))

        #self.backbone.append(nn.Dropout(self.input_dropout))
        self.backbone.append(nn.Linear(128, self.head_input_dim))
        self.normalize_layer = Normalize()

        self.projection_head = BarlowTwinsProjectionHead(
            self.head_input_dim, self.head_hidden_dim, self.head_out_dim
        )
        self.criterion = VICRegLoss(
            lambda_param=self.lambda_param,
            mu_param=self.mu_param,
            nu_param=self.nu_param,
        )

        self.save_hyperparameters()

    def forward(self, x, return_embedding=False):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.normalize_layer(x)
        z = self.projection_head(x)

        if not return_embedding:
            return z
        else:
            return z, x

    def training_step(self, batch, batch_index):
        x0, x1 = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        losses = self.criterion(z0, z1)

        inv_loss, var_loss, cov_loss = losses
        summed_loss = sum(losses)

        self.log_dict(
            {
                "Train/Loss": summed_loss,
                "Train/Invariance_Loss": inv_loss,
                "Train/Variance_Loss": var_loss,
                "Train/Covariance_Loss": cov_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            batch_size=self.batch_size,
        )

        if not self.was_vis:
            x0_grid = torchvision.utils.make_grid(x0)
            x1_grid = torchvision.utils.make_grid(x1)
            self.logger.log_image(images=[x0_grid], key="x0_grid")
            self.logger.log_image(images=[x1_grid], key="x1_grid")
            self.was_vis = True

        return summed_loss

    def validation_step(self, batch, batch_index):
        x0, x1 = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        losses = self.criterion(z0, z1)

        inv_loss, var_loss, cov_loss = losses
        summed_loss = sum(losses)

        self.log_dict(
            {
                "Val/Loss": summed_loss,
                "Val/Invariance_Loss": inv_loss,
                "Val/Variance_Loss": var_loss,
                "Val/Covariance_Loss": cov_loss,
            },
            on_epoch=True,

            prog_bar=True,
            logger=True,
            sync_dist=False,
            batch_size=self.batch_size,
        )

        return summed_loss

    def test_step(self, batch, batch_index):
        img, _ = batch
        z, embeds = self.forward(img, return_embedding=True)
        embeds = embeds.squeeze().detach().cpu().tolist()
        self.test_embeds.append(embeds)

    def on_test_epoch_end(self):
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.test_embeds = np.array(self.test_embeds)

        with open(f"latents/{dt_string}_latents.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.test_embeds)

    def configure_optimizers(self):
        optim = self.optimizer.optim_partial(self.parameters())
        sched = self.optimizer.lr_sched_partial(optim)

        return {"optimizer": optim, "lr_scheduler": sched}
