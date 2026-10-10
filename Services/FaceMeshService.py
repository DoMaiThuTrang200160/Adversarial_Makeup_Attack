import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceMeshService:

    def __init__(self):

        base_options = python.BaseOptions(
            model_asset_path=
            "Assets/face_landmarker.task"
        )

        options = (
            vision.FaceLandmarkerOptions(
                base_options=
                base_options,

                num_faces=1,

                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,

                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
        )

        self.detector = (
            vision.FaceLandmarker
            .create_from_options(
                options
            )
        )

    # ======================================================
    # DETECT
    # ======================================================

    def process(
            self,
            frame_rgb
    ):

        mp_image = mp.Image(
            image_format=
            mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = (
            self.detector
            .detect(mp_image)
        )

        return result

    # ======================================================
    # FACE CROP
    # ======================================================

    def extract_face_crop(
            self,
            frame_bgr,
            padding=0.25
    ):

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        result = self.process(
            frame_rgb
        )

        if (
            not result.face_landmarks
            or len(
                result.face_landmarks
            ) == 0
        ):
            return None

        landmarks = (
            result.face_landmarks[0]
        )

        h, w = (
            frame_bgr.shape[:2]
        )

        xs = []
        ys = []

        for lm in landmarks:

            xs.append(
                lm.x * w
            )

            ys.append(
                lm.y * h
            )

        x_min = min(xs)
        y_min = min(ys)

        x_max = max(xs)
        y_max = max(ys)

        face_w = (
            x_max - x_min
        )

        face_h = (
            y_max - y_min
        )

        # ==========================
        # PADDING
        # ==========================

        pad_x = (
            face_w * padding
        )

        pad_y = (
            face_h * padding
        )

        x1 = int(
            max(
                0,
                x_min - pad_x
            )
        )

        y1 = int(
            max(
                0,
                y_min - pad_y
            )
        )

        x2 = int(
            min(
                w,
                x_max + pad_x
            )
        )

        y2 = int(
            min(
                h,
                y_max + pad_y
            )
        )

        face_crop = (
            frame_bgr[
                y1:y2,
                x1:x2
            ]
        )

        if (
            face_crop.size == 0
        ):
            return None

        return face_crop