import os
import glob
import random
import threading

import cv2
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


_ASSETS_DIR = os.path.join("Assets", "MakeUpStyles")
_MODEL_DIR  = os.path.join("Assets", "BeautyGan")
_IMG_SIZE   = 256


class MakeupRendererService:

    def __init__(self, intensity: float = 0.90):
        self._intensity = intensity

        self._sess      = None
        self._tensor_X  = None
        self._tensor_Y  = None
        self._tensor_Xs = None

        self._makeup_pool: list[tuple[str, np.ndarray]] = []
        self._current_ref_batch: np.ndarray | None = None
        self._current_ref_name:  str = "—"

        # reusable (1, 256, 256, 3) source buffer — avoids malloc per frame
        self._X_buf: np.ndarray | None = None

        self._pending_crop:  np.ndarray | None = None
        self._pending_box:   tuple | None = None
        self._pending_ref:   np.ndarray | None = None
        self._last_result:   np.ndarray | None = None  # (crop_h, crop_w, 3) BGR

        self._lock    = threading.Lock()
        self._event   = threading.Event()
        self._running = False

        self._load_model(_MODEL_DIR)
        self._load_makeup_pool(_ASSETS_DIR)
        self._warmup()
        self._start_worker()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def apply_makeup(
        self,
        frame_bgr: np.ndarray,
        face_crop: np.ndarray,
        face_box:  tuple,           # (x1, y1, x2, y2)
    ) -> np.ndarray:
        """
        Non-blocking. Submits face_crop to the GAN worker and returns
        frame_bgr composited with the latest available GAN result.
        On the first call (no result yet) returns frame_bgr unmodified.
        """
        ref = self._pick_reference()

        with self._lock:
            self._pending_crop = face_crop
            self._pending_box  = face_box
            self._pending_ref  = ref
        self._event.set()

        with self._lock:
            last = self._last_result
            last_box = self._last_box if hasattr(self, "_last_box") else None

        if last is None or last_box is None:
            return frame_bgr

        return self._paste(frame_bgr, last, last_box)

    def randomize_reference(self) -> None:
        name, ref_batch = random.choice(self._makeup_pool)
        self._current_ref_batch = ref_batch
        self._current_ref_name  = name

    @property
    def current_ref_name(self) -> str:
        return self._current_ref_name

    def close(self) -> None:
        self._running = False
        self._event.set()
        if self._sess:
            self._sess.close()
            self._sess = None

    # ==========================================================
    # WORKER THREAD
    # ==========================================================

    def _start_worker(self) -> None:
        self._running = True
        threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="gan-worker",
        ).start()

    def _worker_loop(self) -> None:
        while self._running:
            self._event.wait()
            self._event.clear()

            if not self._running:
                break

            with self._lock:
                crop = self._pending_crop
                box  = self._pending_box
                ref  = self._pending_ref
                self._pending_crop = None
                self._pending_box  = None
                self._pending_ref  = None

            if crop is None or ref is None:
                continue

            result = self._run_gan(crop, ref)

            with self._lock:
                self._last_result = result
                self._last_box    = box

    # ==========================================================
    # MODEL
    # ==========================================================

    def _load_model(self, model_dir: str) -> None:
        meta = os.path.join(model_dir, "model.meta")

        if not os.path.exists(meta):
            raise FileNotFoundError(
                f"BeautyGAN model not found: {meta} — "
                f"place model.meta + checkpoint in {model_dir}"
            )

        tf.reset_default_graph()

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
        )
        config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1
        )

        self._sess = tf.Session(config=config)
        self._sess.run(tf.global_variables_initializer())

        saver = tf.train.import_meta_graph(meta)
        saver.restore(self._sess, tf.train.latest_checkpoint(model_dir))

        graph = tf.get_default_graph()
        self._tensor_X  = graph.get_tensor_by_name("X:0")
        self._tensor_Y  = graph.get_tensor_by_name("Y:0")
        self._tensor_Xs = graph.get_tensor_by_name("generator/xs:0")

        self._X_buf = np.empty((1, _IMG_SIZE, _IMG_SIZE, 3), dtype=np.float32)

    # ==========================================================
    # REFERENCE POOL
    # ==========================================================

    def _load_makeup_pool(self, assets_dir: str) -> None:
        paths = glob.glob(os.path.join(assets_dir, "*.*"))

        if not paths:
            raise FileNotFoundError(
                f"No makeup reference images found in: {assets_dir}"
            )

        self._makeup_pool = [
            (
                os.path.splitext(os.path.basename(p))[0],
                np.expand_dims(
                    self._preprocess(
                        cv2.cvtColor(
                            cv2.resize(self._read_bgr(p), (_IMG_SIZE, _IMG_SIZE)),
                            cv2.COLOR_BGR2RGB,
                        ).astype(np.float32)
                    ),
                    axis=0,
                ),
            )
            for p in sorted(paths)
        ]

    def _pick_reference(self) -> np.ndarray:
        if self._current_ref_batch is None:
            self.randomize_reference()
        return self._current_ref_batch

    # ==========================================================
    # WARMUP
    # ==========================================================

    def _warmup(self) -> None:
        # triggers CUDA kernel compilation at init, not on frame-1
        dummy = np.zeros((1, _IMG_SIZE, _IMG_SIZE, 3), dtype=np.float32)
        self._sess.run(
            self._tensor_Xs,
            feed_dict={self._tensor_X: dummy, self._tensor_Y: dummy},
        )

    # ==========================================================
    # GAN INFERENCE  (worker thread only)
    # ==========================================================

    def _run_gan(self, crop_bgr: np.ndarray, ref_batch: np.ndarray) -> np.ndarray:
        crop_h, crop_w = crop_bgr.shape[:2]

        # source: BGR crop → resize → RGB → preprocess → into reusable buffer
        self._X_buf[0] = self._preprocess(
            cv2.cvtColor(
                cv2.resize(crop_bgr, (_IMG_SIZE, _IMG_SIZE)),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)
        )

        Xs_ = self._sess.run(
            self._tensor_Xs,
            feed_dict={self._tensor_X: self._X_buf, self._tensor_Y: ref_batch},
        )

        result_rgb = (self._deprocess(Xs_[0]) * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        # resize back to original crop dimensions
        return cv2.resize(result_bgr, (crop_w, crop_h))

    # ==========================================================
    # COMPOSITE
    # ==========================================================

    @staticmethod
    def _paste(
        frame: np.ndarray,
        gan_crop: np.ndarray,
        box: tuple,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box
        out = frame.copy()
        out[y1:y2, x1:x2] = gan_crop
        return out

    # ==========================================================
    # UTILITIES
    # ==========================================================

    @staticmethod
    def _preprocess(img_rgb: np.ndarray) -> np.ndarray:
        return (img_rgb / 255.0 - 0.5) * 2.0

    @staticmethod
    def _deprocess(arr: np.ndarray) -> np.ndarray:
        return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _read_bgr(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read: {path}")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img