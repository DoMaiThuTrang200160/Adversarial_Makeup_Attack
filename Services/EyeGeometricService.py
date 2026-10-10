import cv2
import numpy as np

from Utils.ValueObjsType1 import AffineTransform, BoundingBox, EyeGeometry, NormalizedEye

class EyeGeometricService:

    @staticmethod
    def landmarks_to_points(
        landmarks,
        indices,
        img_w,
        img_h
    ):
        pts = []

        for i in indices:
            if i >= len(landmarks):
                continue

            pts.append([
                landmarks[i].x * img_w,
                landmarks[i].y * img_h
            ])

        return np.array(pts, dtype=np.float32)

    @staticmethod
    def compute_geometry(pts):

        center = pts.mean(axis=0)

        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        width = x_max - x_min
        height = y_max - y_min

        return EyeGeometry(
            center=center,
            width=width,
            height=height,
            aspect_ratio=float(height / (width + 1e-6))
        )

    @staticmethod
    def get_eye_box(
        pts,
        img_w,
        img_h
    ):
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        w = x_max - x_min
        h = y_max - y_min

        expand_x = int(w * 0.5)
        expand_y = int(h * 1.0)

        x1 = int(max(0, x_min - expand_x))
        y1 = int(max(0, y_min - expand_y * 0.6))
        x2 = int(min(img_w - 1, x_max + expand_x))
        y2 = int(min(img_h - 1, y_max + expand_y))

        return BoundingBox(x1, y1, x2, y2)

    @staticmethod
    def polygon_mask(pts, box):
        x1, y1, x2, y2 = box.as_tuple()

        w = x2 - x1
        h = y2 - y1

        # translate landmarks into local box space
        pts_local = pts.copy().astype(np.float32)
        pts_local[:, 0] -= x1
        pts_local[:, 1] -= y1

        hull = cv2.convexHull(pts_local.astype(np.int32))

        base = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(base, [hull], 1.0)

        # ── outer layer: dilate rộng + shift xuống → vùng phấn mờ ───────────────
        dilate_px = max(21, int(min(w, h) * 0.18)) | 1
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        outer = cv2.dilate(base, k_dilate)

        shift = int(h * 0.12)
        M = np.float32([[1, 0, 0], [0, 1, shift]])
        outer = np.maximum(outer, cv2.warpAffine(outer, M, (w, h)))

        blur_outer = max(7, (dilate_px // 3) | 1)
        outer = cv2.GaussianBlur(outer, (blur_outer, blur_outer), 0)

        # ── inner ring: sát hull, đậm hơn → vùng kẻ mắt ─────────────────────────
        erode_px = max(3, int(min(w, h) * 0.06)) | 1
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))

        # inner = hull gốc trừ đi phần erode sâu vào trong → chỉ còn viền sát mắt
        inner = base - cv2.erode(base, k_erode)
        inner = np.clip(inner, 0, 1)

        blur_inner = max(3, (erode_px // 2) | 1)
        inner = cv2.GaussianBlur(inner, (blur_inner, blur_inner), 0)

        # ── composite: outer mờ + inner ring đậm thêm ────────────────────────────
        mask = np.clip(outer * 0.6 + inner * 1.0, 0, 1)

        return mask

    @staticmethod
    def normalize(
        image,
        box,
        output_size
    ):

        x1, y1, x2, y2 = box.as_tuple()

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        h, w = crop.shape[:2]

        sx = output_size / w
        sy = output_size / h

        forward = np.array([
            [sx, 0, -x1 * sx],
            [0, sy, -y1 * sy]
        ], dtype=np.float32)

        inverse = np.array([
            [1 / sx, 0, x1],
            [0, 1 / sy, y1]
        ], dtype=np.float32)

        aligned = cv2.resize(
            crop,
            (output_size, output_size)
        )

        transform = AffineTransform(
            forward,
            inverse
        )

        return NormalizedEye(
            image=aligned,
            transform=transform,
            box=box
        )