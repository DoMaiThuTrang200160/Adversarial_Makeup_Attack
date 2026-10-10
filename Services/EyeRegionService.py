import cv2
import numpy as np

from config import EYE_OUTPUT_SIZE
from Utils.ValueObjsType1 import EyeLandMark, EyeRegion, EyeSide
from Services.EyeGeometricService import EyeGeometricService


class EyeRegionService:

    def extract(self, frame_bgr, landmarks):
        h, w = frame_bgr.shape[:2]

        left_eye  = self._extract_eye(frame_bgr, landmarks, EyeLandMark.LEFT_EYE_FULL,  EyeLandMark.LEFT_EYEBALL,  w, h, EyeSide.LEFT)
        right_eye = self._extract_eye(frame_bgr, landmarks, EyeLandMark.RIGHT_EYE_FULL, EyeLandMark.RIGHT_EYEBALL, w, h, EyeSide.RIGHT)

        return left_eye, right_eye

    def _extract_eye(self, image, landmarks, eye_indices, eyeball_indices, img_w, img_h, side):
        pts = EyeGeometricService.landmarks_to_points(landmarks, eye_indices, img_w, img_h)

        if len(pts) < 3:
            return None

        geometry   = EyeGeometricService.compute_geometry(pts)
        box        = EyeGeometricService.get_eye_box(pts, img_w, img_h)
        mask       = EyeGeometricService.polygon_mask(pts, box)
        normalized = EyeGeometricService.normalize(image, box, EYE_OUTPUT_SIZE)
        eye_region = EyeRegion(side=side, points=pts, geometry=geometry, box=box, mask=mask, normalized=normalized)
        eyeball    = self._extract_eyeball(landmarks, eyeball_indices, img_w, img_h)

        return self._remove_eyeball(eye_region, eyeball)

    @staticmethod
    def _extract_eyeball(landmarks, indices, w, h):
        pts  = EyeGeometricService.landmarks_to_points(landmarks, indices, w, h)
        box  = EyeGeometricService.get_eye_box(pts, w, h)
        mask = EyeGeometricService.polygon_mask(pts, box)

        return EyeRegion(side=None, points=pts, geometry=None, box=box, mask=mask)

    @staticmethod
    def _remove_eyeball(eye_region, eyeball_region):
        outer_mask  = eye_region.mask
        inner_small = eyeball_region.mask
        H, W = outer_mask.shape

        inner_mask = np.zeros((H, W), dtype=np.float32)

        dx = int(eyeball_region.box.x1 - eye_region.box.x1)
        dy = int(eyeball_region.box.y1 - eye_region.box.y1)

        h, w = inner_small.shape

        x1  = max(0, dx)
        y1  = max(0, dy)
        x2  = min(W, dx + w)
        y2  = min(H, dy + h)
        sx1 = max(0, -dx)
        sy1 = max(0, -dy)
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        inner_mask[y1:y2, x1:x2] = inner_small[sy1:sy2, sx1:sx2]

        eye_region.mask = cv2.GaussianBlur(
            np.clip(outer_mask - inner_mask, 0, 1),
            (7, 7),
            0,
        )

        return eye_region