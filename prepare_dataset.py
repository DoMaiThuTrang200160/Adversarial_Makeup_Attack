import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import mediapipe as mp

from Models.FaceNetWrapper import FaceNetWrapper
from Core.EyeDetectorService import EyeDetectorService
from Utils.GetDevice import get_device


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

FACE_DIR = os.path.join(ROOT, "AdvMakeup", "faces")
EMB_DIR = os.path.join(ROOT, "AdvMakeup","cache", "embeddings")
MASK_DIR = os.path.join(ROOT, "AdvMakeup","cache", "masks")

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

DEVICE = get_device()

print("[INIT] Loading FaceNet...")
facenet = FaceNetWrapper(device=DEVICE).eval()

print("[INIT] Loading MediaPipe FaceMesh...")
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

eye_detector = EyeDetectorService(face_mesh)



def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def to_tensor(img):
    img = cv2.resize(img, (160, 160))
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0).to(DEVICE)


def save_mask(mask, path):
    mask = np.clip(mask, 0, 1)
    mask = (mask * 255).astype(np.uint8)
    Image.fromarray(mask).save(path)

def process_person(person_id):

    person_dir = os.path.join(FACE_DIR, person_id)
    out_mask_dir = os.path.join(MASK_DIR, person_id)

    os.makedirs(out_mask_dir, exist_ok=True)

    emb_list = []

    images = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(images) == 0:
        print(f"no images: {person_id}")
        return

    for img_name in tqdm(images, desc=f"[{person_id}]"):

        path = os.path.join(person_dir, img_name)

        try:
            img = load_image(path)

            with torch.no_grad():
                tensor = to_tensor(img)
                emb = facenet(tensor, detach=True).cpu()
                emb_list.append(emb)

            result = eye_detector.detect(img)

            if result is None:
                continue

            left_eye, right_eye = result

            mask_full = np.zeros(img.shape[:2], dtype=np.float32)

            for eye in [left_eye, right_eye]:
                box = eye.box
                mask = eye.mask

                x1, y1 = box.x1, box.y1
                h, w = mask.shape

                y2 = min(y1 + h, mask_full.shape[0])
                x2 = min(x1 + w, mask_full.shape[1])

                mh = y2 - y1
                mw = x2 - x1

                if mh > 0 and mw > 0:
                    mask_full[y1:y2, x1:x2] = np.maximum(
                        mask_full[y1:y2, x1:x2],
                        mask[:mh, :mw]
                    )

            mask_full = cv2.resize(mask_full, (160, 160))

            # save mask
            base = os.path.splitext(img_name)[0]
            save_path = os.path.join(out_mask_dir, base + "_mask.png")
            save_mask(mask_full, save_path)

        except Exception as e:
            print(f"error {img_name}: {e}")
            continue

    if len(emb_list) > 0:
        emb_stack = torch.cat(emb_list, dim=0)

        emb_stack = torch.nn.functional.normalize(emb_stack, dim=1)

        emb_mean = emb_stack.mean(dim=0, keepdim=True)
        emb_mean = torch.nn.functional.normalize(emb_mean, dim=1)

        emb_path = os.path.join(EMB_DIR, f"{person_id}.pt")
        torch.save(emb_mean.squeeze(0), emb_path)

        print(f"saved embedding: {person_id}")

def main():

    persons = [
        p for p in os.listdir(FACE_DIR)
        if os.path.isdir(os.path.join(FACE_DIR, p))
    ]

    print(f"[DATASET] persons: {len(persons)}")

    for p in persons:
        process_person(p)

    print("DONE")


if __name__ == "__main__":
    main()