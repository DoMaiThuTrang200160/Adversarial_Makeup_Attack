import os
import random
import torch
from PIL import Image
import torchvision.transforms as T


class AttackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        max_persons=None,
        max_imgs_per_person=None,
        max_makeups=None,
        debug=True
    ):
        self.debug = debug

        self.face_root = os.path.join(root, "faces")
        self.emb_root = os.path.join(root, "cache", "embeddings")
        self.mask_root = os.path.join(root, "cache", "masks")
        self.makeup_root = os.path.join(root, "makeups")

        # ===== CHECK ROOT =====
        for p in [self.face_root, self.emb_root, self.mask_root, self.makeup_root]:
            if not os.path.exists(p):
                raise RuntimeError(f"Missing folder: {p}")

        # ===== load persons =====
        persons = [
            p for p in os.listdir(self.face_root)
            if os.path.isdir(os.path.join(self.face_root, p))
        ]

        if len(persons) == 0:
            raise RuntimeError("No persons found in faces/")

        if max_persons is not None:
            persons = random.sample(persons, min(max_persons, len(persons)))

        self.persons = persons

        self.data = []
        for p in self.persons:
            folder = os.path.join(self.face_root, p)
            imgs = os.listdir(folder)

            if max_imgs_per_person is not None:
                imgs = random.sample(imgs, min(max_imgs_per_person, len(imgs)))

            for img in imgs:
                self.data.append((p, img))

        if len(self.data) == 0:
            raise RuntimeError("No images loaded")

        # ===== load makeup images =====
        self.makeups = [
            f for f in os.listdir(self.makeup_root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if len(self.makeups) == 0:
            raise RuntimeError("No makeup images found")

        if max_makeups is not None:
            self.makeups = random.sample(
                self.makeups,
                min(max_makeups, len(self.makeups))
            )

        print(f"[DATASET] Persons: {len(self.persons)}")
        print(f"[DATASET] Images: {len(self.data)}")
        print(f"[DATASET] Makeups: {len(self.makeups)}")

        # ===== transforms =====
        self.img_transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor()   # [0,1]
        ])

        self.mask_transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor()
        ])

        self.fixed_victim = random.choice(self.persons)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                attacker_id, attacker_img_name = self.data[idx]

                # ===== attacker =====
                attacker_path = os.path.join(
                    self.face_root,
                    attacker_id,
                    attacker_img_name
                )

                attacker_img = Image.open(attacker_path).convert("RGB")

                # ===== victim =====
                victim_id = self.fixed_victim

                if victim_id == attacker_id:
                    idx = random.randint(0, len(self.data) - 1)
                    continue

                victim_emb_path = os.path.join(
                    self.emb_root,
                    f"{victim_id}.pt"
                )

                if not os.path.exists(victim_emb_path):
                    if self.debug:
                        print(f"missing emb: {victim_emb_path}")
                    idx = random.randint(0, len(self.data) - 1)
                    continue

                victim_emb = torch.load(victim_emb_path)

                # ===== mask =====
                base = os.path.splitext(attacker_img_name)[0]
                mask_name = base + "_mask.png"

                mask_path = os.path.join(
                    self.mask_root,
                    attacker_id,
                    mask_name
                )

                if not os.path.exists(mask_path):
                    if self.debug:
                        print(f"missing mask: {mask_path}")
                    idx = random.randint(0, len(self.data) - 1)
                    continue

                mask = Image.open(mask_path).convert("L")

                # ===== makeup =====
                makeup_name = random.choice(self.makeups)

                makeup_path = os.path.join(
                    self.makeup_root,
                    makeup_name
                )

                makeup_img = Image.open(makeup_path).convert("RGB")

                # ===== transform =====
                attacker_img = self.img_transform(attacker_img)
                makeup_img = self.img_transform(makeup_img)

                mask = self.mask_transform(mask)
                mask = torch.clamp(mask, 0.0, 1.0)

                return attacker_img, mask, victim_emb, makeup_img

            except Exception as e:
                if self.debug:
                    print(f"dataset error: {e}")
                idx = random.randint(0, len(self.data) - 1)

        raise RuntimeError(" Too many dataset failures")