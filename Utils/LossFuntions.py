import torch
import torch.nn.functional as F
import torch.nn as nn


def identity_loss(emb_adv, emb_victim, emb_attacker, margin=0.3):

    cos_v = F.cosine_similarity(emb_adv, emb_victim)
    cos_a = F.cosine_similarity(emb_adv, emb_attacker)
    loss_v = F.relu(0.6 - cos_v) ** 2
    loss_a = F.relu(cos_a - margin) ** 2

    return (loss_v + 0.3 * loss_a).mean()

def region_loss(x_adv, x, mask):
    mask = mask.expand_as(x_adv)

    diff = ((1 - mask) * (x_adv - x)).abs()

    denom = torch.clamp((1 - mask).sum(), min=1.0)
    return diff.sum() / denom


def l2_loss(x_adv, x):
    return (x_adv - x).pow(2).mean()


def gan_d_loss(real, fake):
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()

def gan_g_loss(fake):
    return -fake.mean()

def gram_matrix(x):
    B, C, H, W = x.size()

    f = x.view(B, C, H * W)

    # normalize theo feature vector
    f = f / (f.norm(dim=2, keepdim=True) + 1e-6)

    gm = torch.bmm(f, f.transpose(1, 2))

    gm = gm / (C + 1e-6)
    gm = torch.clamp(gm, -1.0, 1.0)

    return gm

def style_loss(vgg, x_adv, makeup):

    x_adv = torch.clamp(x_adv, 0, 1)
    makeup = torch.clamp(makeup, 0, 1)

    x_adv = vgg.preprocess_vgg(x_adv)
    makeup = vgg.preprocess_vgg(makeup)

    feat_adv = vgg(x_adv)
    feat_make = vgg(makeup)

    loss = 0.0

    for fa, fm in zip(feat_adv, feat_make):
        gm_adv = gram_matrix(fa)
        gm_make = gram_matrix(fm)

        diff = (gm_adv - gm_make).abs().mean()
        diff = torch.clamp(diff, 0, 10)

        loss += diff

    return loss / len(feat_adv)

class LaplacianFilter(nn.Module):
    def __init__(self):
        super().__init__()

        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32)

        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1)


def edge_loss(lap, x_adv, makeup, mask):
    x_adv_gray = x_adv.mean(dim=1, keepdim=True)
    makeup_gray = makeup.mean(dim=1, keepdim=True)

    edge_adv = lap(x_adv_gray)
    edge_make = lap(makeup_gray)

    mask = mask.mean(dim=1, keepdim=True)

    diff = (edge_adv - edge_make).abs() * mask

    denom = torch.clamp(mask.sum(), min=1.0)
    return diff.sum() / denom

def smooth_loss(x, mask=None):
    dx = (x[:, :, :-1, :] - x[:, :, 1:, :]).abs()
    dy = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs()

    if mask is not None:
        mask_x = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        mask_y = mask[:, :, :, :-1] * mask[:, :, :, 1:]

        dx = dx * mask_x
        dy = dy * mask_y

    return dx.mean() + dy.mean()


def perceptual_loss(vgg, x_adv, x):

    x_adv = torch.clamp(x_adv, 0, 1)
    x = torch.clamp(x, 0, 1)

    x_adv = vgg.preprocess_vgg(x_adv)
    x = vgg.preprocess_vgg(x)

    feat_adv = vgg(x_adv)
    feat_x = vgg(x)

    loss = 0.0

    for fa, fx in zip(feat_adv, feat_x):
        diff = (fa - fx).abs().mean()
        diff = torch.clamp(diff, 0, 10)

        loss += diff

    return loss / len(feat_adv)


def total_loss(
    loss_id,
    loss_perc,
    loss_reg,
    loss_gan,
    loss_style,
    loss_edge,
    loss_smooth,
    loss_l2,
    w=None
):

    if w is None:
        w = dict(
            id=25.0,
            perc=0.1,
            reg=0.2,
            gan=0.2,
            style=0.2,
            edge=0.1,
            smooth=0.05,
            l2=0.3
        )

    return (
        w["id"] * loss_id +
        w["perc"] * loss_perc +
        w["reg"] * loss_reg +
        w["gan"] * loss_gan +
        w["style"] * loss_style +
        w["edge"] * loss_edge +
        w["smooth"] * loss_smooth +
        w["l2"] * loss_l2
    )