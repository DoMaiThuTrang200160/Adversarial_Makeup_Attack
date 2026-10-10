import cv2
import torch
import numpy as np

from config import (
    DEVICE,
    SIMILARITY_THRESHOLD
)

from Services.DatabaseService import (
    DatabaseService
)

from Services.FaceMeshService import (
    FaceMeshService
)

from Utils.ValueObjsType2 import (
    RecognitionResult
)

from Models.FaceNetWrapper import (
    FaceNetWrapper
)


class FaceRecognitionService:

    def __init__(self):

        self.device = DEVICE

        print(
            f"[Recognition] "
            f"Device: {self.device}"
        )

        self.model = (
            FaceNetWrapper(
                device=self.device
            )
            .eval()
        )

        self.database_service = (
            DatabaseService()
        )

        self.face_mesh_service = (
            FaceMeshService()
        )

        # identity -> list[embedding]
        self.identity_db = {}

        self._build_database()

    # ==========================================================
    # DATABASE BUILD
    # ==========================================================

    def _build_database(self):

        print(
            "[Recognition] "
            "Building database..."
        )

        people = (
            self.database_service
            .get_all_people()
        )

        for person in people:

            embeddings = []

            image_paths = (
                self.database_service
                .get_person_image_paths(
                    person
                )
            )

            for path in image_paths:

                img_rgb = (
                    self.database_service
                    .load_image_rgb(path)
                )

                if img_rgb is None:
                    continue

                # RGB -> BGR
                img_bgr = cv2.cvtColor(
                    img_rgb,
                    cv2.COLOR_RGB2BGR
                )

                # ==================================
                # FACE DETECTION
                # DB PIPELINE == RUNTIME PIPELINE
                # ==================================

                face_crop = (
                    self.face_mesh_service
                    .extract_face_crop(
                        img_bgr
                    )
                )

                if face_crop is None:

                    print(
                        f"[DB] "
                        f"Skip (No Face): "
                        f"{path}"
                    )

                    continue

                emb = (
                    self.extract_embedding(
                        face_crop
                    )
                )

                if emb is not None:
                    embeddings.append(
                        emb
                    )

            if len(embeddings) == 0:

                print(
                    f"[DB] "
                    f"{person}: "
                    f"0 valid images"
                )

                continue

            # store ALL embeddings
            self.identity_db[
                person
            ] = embeddings

            print(
                f"[Recognition] "
                f"{person}: "
                f"{len(embeddings)} "
                f"embeddings"
            )

        print(
            "[Recognition] "
            "Database ready."
        )

    # ==========================================================
    # EMBEDDING
    # ==========================================================

    @torch.no_grad()
    def _embed_rgb(
            self,
            face_rgb: np.ndarray
    ) -> np.ndarray | None:

        if face_rgb is None:
            return None

        face_rgb = cv2.resize(
            face_rgb,
            (160, 160)
        )

        tensor = (
            torch.from_numpy(
                face_rgb.astype(
                    np.float32
                ) / 255.0
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        emb = (
            self.model
            .get_embedding(
                tensor
            )
        )

        return (
            emb[0]
            .cpu()
            .numpy()
        )

    def extract_embedding(
            self,
            face_bgr: np.ndarray
    ) -> np.ndarray | None:

        if face_bgr is None:
            return None

        face_rgb = cv2.cvtColor(
            face_bgr,
            cv2.COLOR_BGR2RGB
        )

        return self._embed_rgb(
            face_rgb
        )

    # ==========================================================
    # RECOGNITION
    # ==========================================================

    def recognize(
            self,
            face_crop_bgr: np.ndarray
    ) -> RecognitionResult:

        if not self.identity_db:
            return (
                RecognitionResult()
            )

        emb = (
            self.extract_embedding(
                face_crop_bgr
            )
        )

        if emb is None:
            return (
                RecognitionResult()
            )

        best_identity = (
            "Unknown"
        )

        best_similarity = -1.0

        # ==================================
        # MAX SIMILARITY
        # ==================================

        for (
            identity,
            db_embeddings
        ) in (
            self.identity_db
            .items()
        ):

            sims = [
                float(
                    np.dot(
                        emb,
                        db_emb
                    )
                )
                for db_emb
                in db_embeddings
            ]

            sim = max(sims)

            print(
                f"[Similarity] "
                f"{identity}: "
                f"{sim:.3f}"
            )

            if sim > best_similarity:

                best_similarity = sim
                best_identity = identity

        recognized = (
            best_similarity >=
            SIMILARITY_THRESHOLD
        )

        return RecognitionResult(
            identity=(
                best_identity
                if recognized
                else "Unknown"
            ),
            similarity=float(
                best_similarity
            ),
            distance=float(
                1.0 -
                best_similarity
            ),
            recognized=recognized
        )

    # ==========================================================
    # REGISTER PERSON
    # ==========================================================

    def register_person(
            self,
            identity: str
    ) -> bool:

        image_paths = (
            self.database_service
            .get_person_image_paths(
                identity
            )
        )

        embeddings = []

        for path in image_paths:

            img_rgb = (
                self.database_service
                .load_image_rgb(path)
            )

            if img_rgb is None:
                continue

            img_bgr = cv2.cvtColor(
                img_rgb,
                cv2.COLOR_RGB2BGR
            )

            face_crop = (
                self.face_mesh_service
                .extract_face_crop(
                    img_bgr
                )
            )

            if face_crop is None:
                continue

            emb = (
                self.extract_embedding(
                    face_crop
                )
            )

            if emb is not None:
                embeddings.append(
                    emb
                )

        if len(embeddings) == 0:
            return False

        self.identity_db[
            identity
        ] = embeddings

        print(
            f"[Recognition] "
            f"Registered: "
            f"{identity} "
            f"({len(embeddings)} "
            f"images)"
        )

        return True