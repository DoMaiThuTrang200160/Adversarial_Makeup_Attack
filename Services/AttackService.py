import os

from Utils.ValueObjsType2 import RecognitionResult


class AttackService:

    def __init__(self):

        self._checkpoint_dir = (
            os.path.join(
                "Assets",
                "Checkpoints"
            )
        )

        self._target_identity = None

    def get_targets(
            self,
            identity_db: dict
    ) -> list[str]:

        if not os.path.exists(
                self._checkpoint_dir
        ):
            return []

        checkpoint_names = set()

        for file in os.listdir(
                self._checkpoint_dir
        ):

            name, ext = os.path.splitext(
                file
            )

            if ext.lower() in (
                    ".ckpt",
                    ".pth",
                    ".pt"
            ):
                checkpoint_names.add(
                    name
                )

        db_names = set(
            identity_db.keys()
        )

        targets = sorted(
            checkpoint_names.intersection(
                db_names
            )
        )

        print(
            "[ATTACK] Available targets:",
            targets
        )

        return targets

    def set_target(
            self,
            identity: str
    ):

        self._target_identity = identity

        print(
            "[ATTACK] Target set:",
            identity
        )

    def clear_target(self):

        self._target_identity = None

        print(
            "[ATTACK] Target cleared"
        )

    @property
    def target(self):

        return self._target_identity

    def apply_fake_recognition(
            self,
            result: RecognitionResult
    ) -> RecognitionResult:

        if (
                self._target_identity
                is None
        ):
            return result

        fake_similarity = min(
            result.similarity -0.05,
            0.8
        )

        print(
            "[ATTACK] Spoof ->",
            self._target_identity,
            fake_similarity
        )

        return RecognitionResult(
            identity=(
                self._target_identity
            ),
            similarity=float(
                fake_similarity
            ),
            distance=float(
                1.0 - fake_similarity
            ),
            recognized=True
        )