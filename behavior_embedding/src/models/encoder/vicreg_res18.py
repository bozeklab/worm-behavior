# https://docs.lightly.ai/self-supervised-learning/examples/vicreg.html
# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
# The projection head is the same as the Barlow Twins one
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


class Normalize(torch.nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)  # L2 normalization


class VICReg(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.was_vis = False

        self.test_embeds = []

        resnet = resnet18(weights=None)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.backbone.append(nn.Flatten())
        self.backbone.append(nn.Linear(512, 256))
        self.backbone.append(nn.BatchNorm1d(256))
        self.backbone.append(nn.LeakyReLU(inplace=True))

        self.backbone.append(nn.Linear(256, 128))
        self.backbone.append(nn.BatchNorm1d(128))
        self.backbone.append(nn.LeakyReLU(inplace=True))

        self.backbone.append(nn.Linear(128, self.input_dim))

        self.save_hyperparameters()

    def forward(self, x):
        out = []
        for batch in x:
            batch_out = self.backbone(batch).flatten(start_dim=1)
            out.append(batch_out)

        stacked_out = torch.stack(out)
        return stacked_out
