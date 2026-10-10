from enum import Enum

class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def as_tuple(self):
        return self.x1, self.y1, self.x2, self.y2


class EyeSide(Enum):
    LEFT = "left"
    RIGHT = "right"

class AffineTransform:
    def __init__(self, forward, inverse):
        self.forward = forward
        self.inverse = inverse

class EyeGeometry:
    def __init__(
        self,
        center,
        width,
        height,
        aspect_ratio
    ):
        self.center = center
        self.width = width
        self.height = height
        self.aspect_ratio = aspect_ratio

    @property
    def openness(self):
        return self.height / (self.width + 1e-6)

class NormalizedEye:
    def __init__(
        self,
        image,
        transform,
        box
    ):
        self.image = image
        self.transform = transform
        self.box = box

class EyeRegion:
    def __init__(
        self,
        side,
        points,
        geometry,
        box,
        mask=None,
        normalized=None
    ):
        self.side = side
        self.points = points
        self.geometry = geometry
        self.box = box
        self.mask = mask
        self.normalized = normalized

class FaceRegion:
    def __init__(
        self,
        box,
        landmarks
    ):
        self.box = box
        self.landmarks = landmarks

class FrameAnalysis:
    def __init__(
        self,
        frame,
        face_region=None,
        left_eye=None,
        right_eye=None
    ):
        self.frame = frame
        self.face_region = face_region
        self.left_eye = left_eye
        self.right_eye = right_eye

class EyeLandMark:
    # LEFT EYE
    # --- Core eye (mí chính) ---
    LEFT_EYE_CORNER = [33, 133]

    LEFT_EYE_UPPER = [159, 158, 157]
    LEFT_EYE_LOWER = [145, 153, 154]

    # --- Outer contour ---
    LEFT_EYE_OUTER = [161, 163]

    # --- Under eye (bọng mắt) ---
    LEFT_UNDER_EYE = [24, 23, 22]

    # --- Eyebrow ---
    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]

    # --- Full (gộp lại nếu cần) ---
    LEFT_EYE_FULL = (
        LEFT_EYE_CORNER +
        LEFT_EYE_UPPER +
        LEFT_EYE_LOWER +
        LEFT_EYE_OUTER +
        LEFT_UNDER_EYE +
        LEFT_EYEBROW
    )

    # RIGHT EYE
    RIGHT_EYE_CORNER = [362, 263]

    RIGHT_EYE_UPPER = [387, 386, 385, 384, 398]
    RIGHT_EYE_LOWER = [373, 374, 380, 381, 382]

    RIGHT_EYE_OUTER = [466, 388, 390, 249]

    RIGHT_UNDER_EYE = [359, 255, 339, 254, 253, 252, 256]

    RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295]

    RIGHT_EYE_FULL = (
        RIGHT_EYE_CORNER +
        RIGHT_EYE_UPPER +
        RIGHT_EYE_LOWER +
        RIGHT_EYE_OUTER +
        RIGHT_UNDER_EYE +
        RIGHT_EYEBROW
    )

    # IRIS
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]


    # EYEBALL (tròng trắng + iris region)
    LEFT_EYEBALL_CORNER = [33, 133]

    LEFT_EYEBALL_UPPER = [160, 159, 158, 157, 173]
    LEFT_EYEBALL_LOWER = [144, 145, 153, 154, 155]

    LEFT_EYEBALL = (
        LEFT_EYEBALL_CORNER +
        LEFT_EYEBALL_UPPER +
        LEFT_EYEBALL_LOWER
    )

    RIGHT_EYEBALL_CORNER = [362, 263]

    RIGHT_EYEBALL_UPPER = [387, 386, 385, 384, 398]
    RIGHT_EYEBALL_LOWER = [373, 374, 380, 381, 382]

    RIGHT_EYEBALL = (
        RIGHT_EYEBALL_CORNER +
        RIGHT_EYEBALL_UPPER +
        RIGHT_EYEBALL_LOWER
    )