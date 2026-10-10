import os
import cv2
import numpy as np

from config import DATABASE_PATH


class DatabaseService:

    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(self):
        self.database_path = DATABASE_PATH
        os.makedirs(self.database_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Người dùng                                                          #
    # ------------------------------------------------------------------ #

    def get_all_people(self) -> list[str]:
        """Trả về danh sách tên (tên folder) trong database."""
        return [
            name
            for name in os.listdir(self.database_path)
            if os.path.isdir(os.path.join(self.database_path, name))
        ]

    def person_exists(self, identity: str) -> bool:
        return os.path.isdir(
            os.path.join(self.database_path, identity)
        )

    def create_person_dir(self, identity: str) -> str:
        """Tạo folder cho người mới, trả về đường dẫn."""
        person_dir = os.path.join(self.database_path, identity)
        os.makedirs(person_dir, exist_ok=True)
        return person_dir

    # ------------------------------------------------------------------ #
    #  Ảnh                                                                 #
    # ------------------------------------------------------------------ #

    def get_person_image_paths(self, identity: str) -> list[str]:
        person_dir = os.path.join(self.database_path, identity)

        if not os.path.isdir(person_dir):
            return []

        return [
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith(self.SUPPORTED_EXTENSIONS)
        ]

    @staticmethod
    def load_image_rgb( path: str) -> np.ndarray | None:
        """
        Đọc ảnh và convert sang RGB uint8 (H, W, 3).
        Trả về None nếu file lỗi hoặc không đọc được.
        """
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_person_images(self, identity: str) -> list[np.ndarray]:
        """
        Đọc toàn bộ ảnh hợp lệ của một người.
        Ảnh corrupt sẽ bị bỏ qua, không raise exception.
        """
        images = []
        for path in self.get_person_image_paths(identity):
            img = self.load_image_rgb(path)
            if img is not None:
                images.append(img)
        return images

    def save_image(
        self,
        identity: str,
        image_bgr: np.ndarray,
        filename: str | None = None
    ) -> str:
        """
        Lưu ảnh BGR (từ OpenCV/camera) vào folder của người đó.
        Tự tạo folder nếu chưa có.
        Trả về đường dẫn file đã lưu.
        """
        person_dir = self.create_person_dir(identity)

        if filename is None:
            existing = self.get_person_image_paths(identity)
            filename = f"{len(existing):04d}.jpg"

        save_path = os.path.join(person_dir, filename)
        cv2.imwrite(save_path, image_bgr)
        return save_path

    def count_images(self, identity: str) -> int:
        return len(self.get_person_image_paths(identity))