import cv2
import numpy as np
from datetime import datetime

from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QFrame, QSizePolicy,
    QScrollArea, QGridLayout, QSpacerItem,
)

from config import CAMERA_INDEX, MASK_ALPHA, PREVIEW_SIZE, RECOGNITION_INTERVAL
from Services.FaceMeshService import FaceMeshService
from Services.EyeRegionService import EyeRegionService
from Services.FaceRecognitionService import FaceRecognitionService
from Services.MakeupRendererService import MakeupRendererService
from Services.AttackService import AttackService


# ──────────────────────────────────────────────────────────────
# STYLESHEET
# ──────────────────────────────────────────────────────────────
APP_STYLE = """
/* ── Global ─────────────────────────────────────────────── */
QWidget {
    background-color: #F0F2F5;
    color: #1A1D23;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 12px;
}

/* ── Title Bar ───────────────────────────────────────────── */
#titleBar {
    background-color: #1A1D23;
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}

/* ── Camera Panel ────────────────────────────────────────── */
#cameraPanel {
    background-color: #0D0F12;
    border-radius: 4px;
}
#cameraHeader {
    background-color: #1A1D23;
    border-radius: 4px 4px 0 0;
    padding: 4px 10px;
}
#liveDot {
    color: #00D46A;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
#cameraLabel {
    background-color: #0D0F12;
}

/* ── Right Panel ─────────────────────────────────────────── */
#rightPanel {
    background-color: #FFFFFF;
    border-left: 1px solid #E2E5EA;
}

/* ── Section Cards ───────────────────────────────────────── */
#sectionCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E5EA;
    border-radius: 6px;
    margin: 0px;
}
#sectionHeader {
    background-color: #F7F8FA;
    border-bottom: 1px solid #E2E5EA;
    border-radius: 6px 6px 0 0;
    padding: 6px 12px;
}
#sectionHeaderLabel {
    font-size: 11px;
    font-weight: 700;
    color: #5A6072;
    letter-spacing: 0.8px;
}
#sectionBody {
    padding: 10px 12px;
}

/* ── Recognition Result ──────────────────────────────────── */
#nameLabel {
    font-size: 18px;
    font-weight: 700;
    color: #1A1D23;
}
#statusBadgeOK {
    background-color: #E6F9F0;
    color: #00A854;
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 700;
}
#statusBadgeFail {
    background-color: #FFF0F0;
    color: #E53935;
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 700;
}
#metaKey {
    color: #8A90A0;
    font-size: 12px;
}
#metaValue {
    color: #1A1D23;
    font-size: 12px;
    font-weight: 600;
}

/* ── Embedding text ──────────────────────────────────────── */
#embeddingText {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 10px;
    color: #6B7280;
    background-color: #F7F8FA;
    border-radius: 4px;
    padding: 6px 8px;
}

/* ── Eye Region Rows ─────────────────────────────────────── */
#eyeRowLabel {
    font-size: 12px;
    color: #1A1D23;
}
#eyeRowOK {
    font-size: 11px;
    font-weight: 700;
    color: #00A854;
}
#eyeRowFail {
    font-size: 11px;
    font-weight: 700;
    color: #E53935;
}

/* ── System Info ─────────────────────────────────────────── */
#sysKey {
    color: #8A90A0;
    font-size: 11px;
}
#sysValue {
    color: #1A1D23;
    font-size: 11px;
    font-weight: 600;
}

/* ── Bottom Status Bar ───────────────────────────────────── */
#statusBar {
    background-color: #1A1D23;
    border-top: 1px solid #2A2D35;
}

/* ── Buttons ─────────────────────────────────────────────── */
#btnStart {
    background-color: #1A1D23;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 5px 16px;
    font-size: 12px;
    font-weight: 600;
}
#btnStart:hover {
    background-color: #2E3240;
}
#btnStart:pressed {
    background-color: #0D0F12;
}

#btnAttackOff {
    background-color: transparent;
    color: #8A90A0;
    border: 1px solid #3A3D45;
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
}
#btnAttackOff:hover {
    background-color: #2E3240;
    color: #FFFFFF;
}

#btnAttackOn {
    background-color: #E53935;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
}
#btnAttackOn:hover {
    background-color: #FF5252;
}

#btnViewLog {
    background-color: transparent;
    color: #8A90A0;
    border: 1px solid #3A3D45;
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
}
#btnViewLog:hover {
    background-color: #2E3240;
    color: #FFFFFF;
}

/* ── Combo ───────────────────────────────────────────────── */
QComboBox {
    border: 1px solid #E2E5EA;
    border-radius: 4px;
    padding: 4px 8px;
    background: #FFFFFF;
    font-size: 12px;
    color: #1A1D23;
}
QComboBox:hover { border-color: #A0A8B8; }
QComboBox::drop-down { border: none; width: 20px; }

/* ── Separator ───────────────────────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #E2E5EA;
}
"""


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def _make_section(title_text: str) -> tuple[QFrame, QVBoxLayout]:
    """Returns (card_frame, body_layout) — add content to body_layout."""
    card = QFrame()
    card.setObjectName("sectionCard")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(0, 0, 0, 0)
    card_layout.setSpacing(0)

    header = QFrame()
    header.setObjectName("sectionHeader")
    h_lay = QHBoxLayout(header)
    h_lay.setContentsMargins(12, 6, 12, 6)
    icon_map = {
        "RECOGNITION RESULT": "👤",
        "FACE EMBEDDING (512-D)": "🧠",
        "EYE REGION ANALYSIS": "👁",
        "SYSTEM INFO": "⚙",
    }
    prefix = icon_map.get(title_text, "▪")
    lbl = QLabel(f"  {prefix}  {title_text}")
    lbl.setObjectName("sectionHeaderLabel")
    h_lay.addWidget(lbl)
    h_lay.addStretch()

    body = QFrame()
    body.setObjectName("sectionBody")
    body_layout = QVBoxLayout(body)
    body_layout.setContentsMargins(12, 10, 12, 12)
    body_layout.setSpacing(6)

    card_layout.addWidget(header)
    card_layout.addWidget(body)
    return card, body_layout


def _row(key: str, value: str, key_id="metaKey", val_id="metaValue") -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    k = QLabel(key)
    k.setObjectName(key_id)
    k.setMinimumWidth(80)
    v = QLabel(value)
    v.setObjectName(val_id)
    lay.addWidget(k)
    lay.addWidget(v)
    lay.addStretch()
    return w, v  # return value label so we can update it


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


# ──────────────────────────────────────────────────────────────
# MAIN WINDOW
# ──────────────────────────────────────────────────────────────
class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOV-MAKEUP FACE RECOGNITION SYSTEM")
        self.resize(1280, 800)
        self.setStyleSheet(APP_STYLE)

        # Services
        self.face_mesh_service   = FaceMeshService()
        self.eye_service         = EyeRegionService()
        self.recognition_service = FaceRecognitionService()
        self.makeup_service      = MakeupRendererService()
        self.attack_service      = AttackService()

        self.attack_mode    = False
        self.frame_count    = 0
        self.current_result = None
        self.fps_counter    = 0
        self.fps_value      = 0.0
        self._fps_timer_val = datetime.now()

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._build_ui()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

    # ══════════════════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Title bar ──────────────────────────────────────────
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(32)
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(12, 0, 12, 0)
        tb_title = QLabel("AOV-MAKEUP FACE RECOGNITION SYSTEM")
        tb_title.setObjectName("titleBar")
        tb_title.setStyleSheet("color:#FFFFFF; font-size:11px; font-weight:600; letter-spacing:1px;")
        tb_lay.addWidget(tb_title)
        tb_lay.addStretch()

        root.addWidget(title_bar)

        # ── Body (camera + right panel) ────────────────────────
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        body_lay.addWidget(self._build_camera_panel(), stretch=3)
        body_lay.addWidget(self._build_right_panel(),  stretch=1)

        root.addWidget(body, stretch=1)

        # ── Status bar ─────────────────────────────────────────
        root.addWidget(self._build_status_bar())

    # ── Camera Panel ──────────────────────────────────────────
    def _build_camera_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("cameraPanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # header strip
        header = QFrame()
        header.setObjectName("cameraHeader")
        header.setFixedHeight(28)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(10, 0, 10, 0)
        dot = QLabel("● LIVE CAMERA")
        dot.setObjectName("liveDot")
        h_lay.addWidget(dot)
        h_lay.addStretch()

        # camera image
        self.camera_label = QLabel()
        self.camera_label.setObjectName("cameraLabel")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay.addWidget(header)
        lay.addWidget(self.camera_label, stretch=1)
        return panel

    # ── Right Panel ───────────────────────────────────────────
    def _build_right_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setFixedWidth(320)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: #FFFFFF; }")

        content = QWidget()
        content.setStyleSheet("background: #FFFFFF;")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # ── Recognition Result ──────────────────────────────────
        card_rec, body_rec = _make_section("RECOGNITION RESULT")

        self._name_label = QLabel("—")
        self._name_label.setObjectName("nameLabel")

        name_row = QHBoxLayout()
        name_row.setContentsMargins(0,0,0,0)
        name_row.addWidget(self._name_label)
        name_row.addStretch()

        self._status_badge = QLabel("Unknown")
        self._status_badge.setObjectName("statusBadgeFail")
        name_row.addWidget(self._status_badge)
        body_rec.addLayout(name_row)
        body_rec.addWidget(_separator())

        self._id_val = QLabel("—")
        self._status_val = QLabel("—")
        self._similarity_val = QLabel("—")
        self._distance_val = QLabel("—")
        self._time_val = QLabel("—")

        for lbl in (
                self._id_val,
                self._status_val,
                self._similarity_val,
                self._distance_val,
                self._time_val,
        ):
            lbl.setObjectName("metaValue")

        body_rec.addWidget(self._make_kv_row("ID", self._id_val))
        body_rec.addWidget(self._make_kv_row("Status", self._status_val))
        body_rec.addWidget(self._make_kv_row("Similarity", self._similarity_val))
        body_rec.addWidget(self._make_kv_row("Distance", self._distance_val))
        body_rec.addWidget(self._make_kv_row("Time", self._time_val))

        lay.addWidget(card_rec)

        # ── Face Embedding ──────────────────────────────────────
        card_emb, body_emb = _make_section("FACE EMBEDDING (512-D)")
        self._embedding_label = QLabel("No data")
        self._embedding_label.setObjectName("embeddingText")
        self._embedding_label.setWordWrap(True)
        self._embedding_label.setMaximumHeight(60)
        body_emb.addWidget(self._embedding_label)
        lay.addWidget(card_emb)

        # ── Eye Region Analysis ─────────────────────────────────
        card_eye, body_eye = _make_section("EYE REGION ANALYSIS")

        self._eye_rows = {}
        eye_items = [
            "Eyebrow (Left)", "Eyebag (Left)", "Eye Corner (Left)",
            "Eyebrow (Right)", "Eyebag (Right)", "Eye Corner (Right)",
        ]
        for item in eye_items:
            row_w, status_lbl = self._make_eye_row(item)
            self._eye_rows[item] = status_lbl
            body_eye.addWidget(row_w)

        lay.addWidget(card_eye)

        # ── System Info ─────────────────────────────────────────
        card_sys, body_sys = _make_section("SYSTEM INFO")

        self._sys_model_val  = QLabel("FaceNet (GG)")
        self._sys_input_val  = QLabel("160×160")
        self._sys_device_val = QLabel("GPU (CUDA)")
        self._sys_fps_val    = QLabel("0.0")

        for lbl in (self._sys_model_val, self._sys_input_val,
                    self._sys_device_val, self._sys_fps_val):
            lbl.setObjectName("sysValue")

        body_sys.addWidget(self._make_sys_row("Model",      self._sys_model_val))
        body_sys.addWidget(self._make_sys_row("Input Size", self._sys_input_val))
        body_sys.addWidget(self._make_sys_row("Device",     self._sys_device_val))
        body_sys.addWidget(self._make_sys_row("FPS",        self._sys_fps_val))

        lay.addWidget(card_sys)
        lay.addStretch()

        scroll.setWidget(content)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)
        return panel

    # ── Status Bar ────────────────────────────────────────────
    def _build_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("statusBar")
        bar.setFixedHeight(40)

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(12)

        # Start Camera button
        self.btn_start = QPushButton("▶  Start Camera")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setFixedHeight(28)
        self.btn_start.clicked.connect(self._on_toggle_camera)

        # Attack Mode button
        self.btn_attack = QPushButton("⚡  Attack Mode: OFF")
        self.btn_attack.setObjectName("btnAttackOff")
        self.btn_attack.setFixedHeight(28)
        self.btn_attack.clicked.connect(self._on_toggle_attack)

        # Target combo
        self.target_combo = QComboBox()
        self.target_combo.setFixedHeight(28)
        self.target_combo.addItem("Select Target")
        self.target_combo.addItems(
            self.attack_service.get_targets(self.recognition_service.identity_db)
        )
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        self.target_combo.setStyleSheet(
            "QComboBox { background:#2E3240; color:#CCCCCC; border:1px solid #444; "
            "border-radius:4px; padding:3px 8px; } "
            "QComboBox::drop-down { border:none; }"
        )

        # Database info
        db_lbl = QLabel(f"🗄  Database: {len(self.recognition_service.identity_db)} faces")
        db_lbl.setStyleSheet("color:#8A90A0; font-size:11px;")

        # View Log button
        btn_log = QPushButton("📋  View Log")
        btn_log.setObjectName("btnViewLog")
        btn_log.setFixedHeight(28)

        lay.addWidget(self.btn_start)
        lay.addWidget(self.btn_attack)
        lay.addWidget(self.target_combo)
        lay.addStretch()
        lay.addWidget(db_lbl)
        lay.addWidget(btn_log)

        return bar

    # ── Row helpers ───────────────────────────────────────────
    @staticmethod
    def _make_kv_row(key: str, value_label: QLabel) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(8)
        k = QLabel(key)
        k.setObjectName("metaKey")
        k.setFixedWidth(70)
        lay.addWidget(k)
        lay.addWidget(value_label)
        lay.addStretch()
        return w

    @staticmethod
    def _make_eye_row(name: str) -> tuple[QWidget, QLabel]:
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)
        check = QLabel("✓")
        check.setStyleSheet("color:#00A854; font-weight:700; font-size:12px;")
        check.setFixedWidth(16)
        name_lbl = QLabel(name)
        name_lbl.setObjectName("eyeRowLabel")
        status = QLabel("OK")
        status.setObjectName("eyeRowOK")
        lay.addWidget(check)
        lay.addWidget(name_lbl)
        lay.addStretch()
        lay.addWidget(status)
        return w, status

    @staticmethod
    def _make_sys_row(key: str, value_label: QLabel) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)
        k = QLabel(key)
        k.setObjectName("sysKey")
        k.setFixedWidth(75)
        lay.addWidget(k)
        lay.addWidget(value_label)
        lay.addStretch()
        return w

    # ══════════════════════════════════════════════════════════
    # FRAME LOOP
    # ══════════════════════════════════════════════════════════

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # FPS
        self.fps_counter += 1
        now = datetime.now()
        delta = (now - self._fps_timer_val).total_seconds()
        if delta >= 1.0:
            self.fps_value = self.fps_counter / delta
            self.fps_counter = 0
            self._fps_timer_val = now
            self._sys_fps_val.setText(f"{self.fps_value:.1f}")

        results = self.face_mesh_service.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        if not results.face_landmarks:
            self._show_frame(frame)
            return

        landmarks           = results.face_landmarks[0]
        left_eye, right_eye = self.eye_service.extract(frame, landmarks)
        face_crop, face_box = self._extract_face_crop(frame, landmarks)

        # Attack / Normal
        if self.attack_mode and left_eye and right_eye:
            frame = self.makeup_service.apply_makeup(
                frame_bgr=frame,
                face_crop=face_crop,
                face_box=face_box,
            )
        else:
            if left_eye:
                self._draw_eye_mask(frame, left_eye)
            if right_eye:
                self._draw_eye_mask(frame, right_eye)

        # Face recognition
        self.frame_count += 1
        if self.frame_count % RECOGNITION_INTERVAL == 0:
            result = self.recognition_service.recognize(face_crop)

            if self.attack_mode:
                result = self.attack_service.apply_fake_recognition(result)

            self.current_result = result
            self._update_recognition_panel(result)
            self._update_embedding_panel(face_crop)
            self._time_val.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if left_eye and right_eye:
            self._update_eye_analysis(left_eye, right_eye)

        self._show_frame(frame)

    # ══════════════════════════════════════════════════════════
    # PANEL UPDATERS
    # ══════════════════════════════════════════════════════════

    def _update_recognition_panel(self, result):
        self._name_label.setText(result.identity or "Unknown")
        recognized = result.recognized
        if recognized:
            self._status_badge.setText("Recognized")
            self._status_badge.setObjectName("statusBadgeOK")
        else:
            self._status_badge.setText("Unknown")
            self._status_badge.setObjectName("statusBadgeFail")
        self._status_badge.setStyleSheet("")  # force style refresh
        self._status_badge.setStyleSheet(APP_STYLE)

        self._id_val.setText(result.identity or "—")
        self._status_val.setText("Recognized" if recognized else "Unknown")
        self._status_val.setStyleSheet("color:#00A854; font-weight:600;" if recognized
                                       else "color:#E53935; font-weight:600;")
        self._similarity_val.setText(f"{result.similarity:.2f}" if result.similarity else "—")
        self._distance_val.setText(
            f"{(1 - result.similarity):.2f}" if result.similarity else "—"
        )

    def _update_embedding_panel(self, face_crop):
        """Show first few values of the face embedding vector."""
        try:
            emb = self.recognition_service.get_embedding(face_crop)
            if emb is not None:
                vals = ", ".join(f"{v:.4f}" for v in emb[:8])
                self._embedding_label.setText(f"{vals}, ...\n(512 dimensions)")
        except Exception:
            pass

    def _update_eye_analysis(self, left_eye, right_eye):
        # Update eye region status (OK / geometric check)
        checks = {
            "Eyebrow (Left)":   left_eye.geometry.openness > 0.01,
            "Eyebag (Left)":    left_eye.geometry.height   > 2,
            "Eye Corner (Left)":left_eye.geometry.width    > 10,
            "Eyebrow (Right)":  right_eye.geometry.openness > 0.01,
            "Eyebag (Right)":   right_eye.geometry.height   > 2,
            "Eye Corner (Right)":right_eye.geometry.width   > 10,
        }
        for name, ok in checks.items():
            lbl = self._eye_rows.get(name)
            if lbl:
                lbl.setText("OK" if ok else "—")
                lbl.setObjectName("eyeRowOK" if ok else "eyeRowFail")
                lbl.setStyleSheet("color:#00A854; font-weight:700;" if ok
                                  else "color:#E53935; font-weight:700;")

    # ══════════════════════════════════════════════════════════
    # FACE CROP
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _extract_face_crop(frame, landmarks):
        h, w = frame.shape[:2]
        xs = [int(p.x * w) for p in landmarks]
        ys = [int(p.y * h) for p in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        mx = int((x_max - x_min) * 0.18)
        my = int((y_max - y_min) * 0.22)
        x1 = max(0, x_min - mx)
        y1 = max(0, y_min - my)
        x2 = min(w, x_max + mx)
        y2 = min(h, y_max + my)
        face_crop = frame[y1:y2, x1:x2].copy()
        # Draw green face box (matches reference UI)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return face_crop, (x1, y1, x2, y2)

    # ══════════════════════════════════════════════════════════
    # DRAW MASK
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _draw_eye_mask(frame, eye_region):
        x1, y1, x2, y2 = eye_region.box.as_tuple()
        mask = eye_region.mask
        if mask is None:
            return
        overlay      = frame.copy()
        red_layer    = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        red_layer[:] = (0, 0, 255)
        mask_3d      = np.stack([mask] * 3, axis=-1)
        overlay[y1:y2, x1:x2] = (
            overlay[y1:y2, x1:x2] * (1 - mask_3d * MASK_ALPHA)
            + red_layer * (mask_3d * MASK_ALPHA)
        ).astype(np.uint8)
        frame[:] = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    # ══════════════════════════════════════════════════════════
    # SHOW FRAME
    # ══════════════════════════════════════════════════════════

    def _show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = frame_rgb.shape
        self.camera_label.setPixmap(
            QPixmap.fromImage(
                QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            ).scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # ══════════════════════════════════════════════════════════
    # BUTTON / KEY HANDLERS
    # ══════════════════════════════════════════════════════════

    def _on_toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_start.setText("▶  Start Camera")
        else:
            self.timer.start(16)
            self.btn_start.setText("⏹  Stop Camera")

    def _on_toggle_attack(self):
        self.attack_mode = not self.attack_mode
        if self.attack_mode:
            self.btn_attack.setText("⚡  Attack Mode: ON")
            self.btn_attack.setObjectName("btnAttackOn")
        else:
            self.btn_attack.setText("⚡  Attack Mode: OFF")
            self.btn_attack.setObjectName("btnAttackOff")
        # Force stylesheet re-apply
        self.btn_attack.setStyleSheet("")
        self.btn_attack.setStyleSheet(APP_STYLE)
        print("[ATTACK]", "ON" if self.attack_mode else "OFF")

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self._on_toggle_attack()
        elif key == Qt.Key_R:
            self.makeup_service.randomize_reference()
            print("[MAKEUP]", self.makeup_service.current_ref_name)

    def _on_target_changed(self, target: str):
        if target == "Select Target":
            self.attack_service.clear_target()
        else:
            self.attack_service.set_target(target)

    # ══════════════════════════════════════════════════════════
    # CLOSE
    # ══════════════════════════════════════════════════════════

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        self.makeup_service.close()
        event.accept()