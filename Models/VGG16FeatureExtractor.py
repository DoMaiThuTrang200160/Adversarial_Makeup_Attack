from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from Utils.GetDevice import get_device


VggOutputs = namedtuple(
    "VggOutputs",
    ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
)


class VGG16FeatureExtractor(nn.Module):

    def __init__(self, pretrained=True, device=None, debug=False):
        super().__init__()

        self.debug = debug
        device = device or get_device()

        if pretrained:
            vgg = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1
            ).features
        else:
            vgg = models.vgg16(weights=None).features

        self.slice1 = nn.Sequential(*vgg[:4])
        self.slice2 = nn.Sequential(*vgg[4:9])
        self.slice3 = nn.Sequential(*vgg[9:16])
        self.slice4 = nn.Sequential(*vgg[16:23])

        for p in self.parameters():
            p.requires_grad = False

        self.eval()
        self.to(device)

    def forward(self, x):

        def safe(name, t):
            if self.debug:
                if torch.isnan(t).any():
                    print(f"NaN in {name}")
                if torch.isinf(t).any():
                    print(f"INF in {name}")
                print(f"[{name}] min={t.min().item():.3f} max={t.max().item():.3f}")

            return torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)

        x = safe("input", x)

        h = self.slice1(x)
        relu1_2 = safe("relu1_2", h)

        h = self.slice2(h)
        relu2_2 = safe("relu2_2", h)

        h = self.slice3(h)
        relu3_3 = safe("relu3_3", h)

        h = self.slice4(h)
        relu4_3 = safe("relu4_3", h)

        return VggOutputs(
            relu1_2,
            relu2_2,
            relu3_3,
            relu4_3
        )


    @staticmethod
    def preprocess_vgg(x):

        # detect range
        if x.max() > 1.5:
            x = x / 255.0

        x = torch.clamp(x, 0.0, 1.0)

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device
        ).view(1, 3, 1, 1)

        x = (x - mean) / std

        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        return x