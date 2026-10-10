import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from AttackDataset import AttackDataset
from Utils.LossFuntions import *
from Models.FaceNetWrapper import FaceNetWrapper
from Models.GanNetwork import GanNetwork
from Models.VGG16FeatureExtractor import VGG16FeatureExtractor
from Utils.GetDevice import get_device


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VICTIM_PATH = os.path.join(BASE_DIR, "victims", "target.jpg")

SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = get_device()
torch.backends.cudnn.benchmark = True


def load_victim(path, transform):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)
    print("[VICTIM] Loaded target")
    return img


def has_nan(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item()
    return total


# TRAIN
def train():

    print("RUNNING FILE:", __file__)

    dataset = AttackDataset(DATASET_DIR, max_imgs_per_person=200)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    gan = GanNetwork(DEVICE)
    G, D = gan.generator, gan.discriminator

    fr = FaceNetWrapper(device=DEVICE).eval()
    vgg = VGG16FeatureExtractor().to(DEVICE).eval()

    for p in fr.parameters():
        p.requires_grad = False
    for p in vgg.parameters():
        p.requires_grad = False

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-5, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    scaler_G = torch.amp.GradScaler("cuda")
    scaler_D = torch.amp.GradScaler("cuda")

    lap = LaplacianFilter().to(DEVICE)

    victim_img = load_victim(VICTIM_PATH, dataset.img_transform).to(DEVICE)

    with torch.no_grad():
        victim_emb = fr(victim_img, detach=True)

    # LOOP
    for epoch in range(12):

        print(f"\n===== EPOCH {epoch} =====")
        pbar = tqdm(loader)

        for step, (attacker_img, mask, attacker_emb, makeup_img) in enumerate(pbar):

            attacker_img = attacker_img.to(DEVICE)
            mask = mask.to(DEVICE).float()
            makeup_img = makeup_img.to(DEVICE)
            attacker_emb = F.normalize(attacker_emb.to(DEVICE), dim=1)

            # MASK
            mask = torch.clamp(mask* 2.5, 0, 1)
            victim_batch = victim_emb.repeat(attacker_img.size(0), 1)

            # TRAIN D
            with torch.no_grad():
                fake_img, _ = G(attacker_img, mask, victim_batch, makeup_img)
                fake_img = fake_img.detach()


            with torch.autocast("cuda", enabled=False):
                real_out = D(attacker_img)
                fake_out = D(fake_img)
                loss_D = gan_d_loss(real_out, fake_out)

            if not has_nan(loss_D):
                opt_D.zero_grad(set_to_none=True)
                scaler_D.scale(loss_D).backward()
                scaler_D.step(opt_D)
                scaler_D.update()

            # TRAIN G
            with torch.autocast("cuda", enabled=False):
                adv_img, perturb = G(attacker_img, mask, victim_batch, makeup_img)
                adv_img = torch.clamp(adv_img, 0, 1)
            perturb = torch.clamp(perturb, -0.15, 0.15)

            if has_nan(adv_img):
                print("NaN adv_img → skip")
                continue

            if has_nan(perturb):
                print("NaN perturb → skip")
                continue

            # FaceNet FP32
            with torch.amp.autocast("cuda", enabled=False):
                emb_adv = fr(adv_img.float(), detach=False)
                emb_adv = torch.clamp(emb_adv, -1, 1)
            if has_nan(emb_adv):
                print("NaN embedding → skip")
                continue

            emb_adv = torch.clamp(emb_adv, -1, 1)

            # LOSSES
            loss_id = identity_loss(emb_adv, victim_batch, attacker_emb)
            loss_perc = perceptual_loss(vgg, adv_img, attacker_img)
            loss_reg = region_loss(adv_img, attacker_img, mask)
            loss_smooth = smooth_loss(perturb, mask)
            loss_style = style_loss(vgg, adv_img, makeup_img)
            loss_edge = edge_loss(lap, adv_img, makeup_img, mask)
            loss_l2 = perturb.pow(2).mean()

            with torch.autocast("cuda", enabled=False):
                fake_out = D(adv_img)
                loss_gan = gan_g_loss(fake_out)

                loss_G = total_loss(
                    loss_id, loss_perc, loss_reg,
                    loss_gan, loss_style,
                    loss_edge, loss_smooth, loss_l2
                )

            if has_nan(loss_G) or loss_G.abs() > 1e4:
                print("Bad loss → skip")
                continue

            # BACKWARD
            opt_G.zero_grad(set_to_none=True)

            scaler_G.scale(loss_G).backward()

            # check grad
            grad_nan = False
            for p in G.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        grad_nan = True
                        break

            if grad_nan:
                print("NaN gradient → skip")
                opt_G.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(G.parameters(), 0.2)
            g_norm = grad_norm(G)

            scaler_G.step(opt_G)
            scaler_G.update()

            # LOG
            if step % 20 == 0:

                with torch.no_grad():
                    cos_v = F.cosine_similarity(emb_adv, victim_batch).mean().item()
                    cos_a = F.cosine_similarity(emb_adv, attacker_emb).mean().item()
                    perturb_mean = perturb.abs().mean().item()
                    mask_mean = mask.mean().item()

                pbar.set_postfix({
                    "cos_v": f"{cos_v:.3f}",
                    "cos_a": f"{cos_a:.3f}",
                    "G": f"{loss_G.item():.2f}",
                    "D": f"{loss_D.item():.2f}",
                    "p": f"{perturb_mean:.4f}",
                    "mask": f"{mask_mean:.3f}",
                    "g": f"{g_norm:.2f}"
                })

        torch.save(G.state_dict(), os.path.join(SAVE_DIR, f"G_epoch_{epoch}.pth"))

    print("Training done")


if __name__ == "__main__":
    train()