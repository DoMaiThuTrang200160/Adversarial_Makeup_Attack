import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class FiLM(nn.Module):
    def __init__(self, emb_dim, num_features):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, num_features)
        self.beta = nn.Linear(emb_dim, num_features)

    def forward(self, x, emb):
        gamma = torch.tanh(self.gamma(emb)) * 0.2
        beta = torch.tanh(self.beta(emb)) * 0.1

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * (1 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + 0.1 * self.block(x)

class MakeupEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),

            ResBlock(128),
            ResBlock(128),

            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        feat = self.model(x)
        return feat.view(feat.size(0), -1)


class Generator(nn.Module):
    def __init__(self, emb_dim=512, style_dim=128):
        super().__init__()

        self.epsilon = 40 / 255.0

        self.makeup_encoder = MakeupEncoder()

        self.film_id1 = FiLM(emb_dim, 128)
        self.film_id2 = FiLM(emb_dim, 64)

        self.film_style1 = FiLM(style_dim, 128)
        self.film_style2 = FiLM(style_dim, 64)

        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, 2, 2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.res = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, mask, target_emb, makeup):

        mask = mask[:, :1].clamp(0, 1)

        style_emb = self.makeup_encoder(makeup)

        x_in = torch.cat([x, mask], dim=1)

        x1 = self.enc1(x_in)
        x2 = self.enc2(x1)

        x2 = self.res(x2)

        # Clamp chống explode
        x2 = torch.clamp(x2, -10, 10)

        # Apply FiLM
        x2 = self.film_id1(x2, target_emb)
        x2 = x2 + 0.1 * self.film_style1(x2, style_emb)

        d1 = self.dec1(x2)
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)

        d2 = self.dec2(torch.cat([d1, x1], dim=1))
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)

        d2 = torch.clamp(d2, -10, 10)

        d2 = self.film_id2(d2, target_emb)
        d2 = d2 + 0.1 * self.film_style2(d2, style_emb)

        perturb = self.out_conv(d2)

        perturb = torch.tanh(perturb) * (self.epsilon * 1.2)

        adv = x + perturb * mask
        adv = torch.clamp(adv, 0, 1)

        return adv, perturb


class Discriminator(nn.Module):
    def __init__(self, input_dim=3, ndf=64):
        super().__init__()

        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, stride, 1)),
                nn.LeakyReLU(0.2)
            )

        self.model = nn.Sequential(
            block(input_dim, ndf, 2),
            block(ndf, ndf * 2, 2),
            block(ndf * 2, ndf * 4, 2),
            block(ndf * 4, ndf * 8, 2),
            block(ndf * 8, ndf * 8, 2),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))
        )

    def forward(self, x):
        return self.model(x).view(x.size(0), -1)

class GanNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(
                    m.weight,
                    a=0.2,
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                m.weight.data *= 0.1

                if m.bias is not None:
                    nn.init.zeros_(m.bias)