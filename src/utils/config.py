import json
import math
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
MODEL_FILE = PROJECT_ROOT / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Runtime tuning defaults
PROCESS_W, PROCESS_H = 640, 480
PINCH_THRESHOLD = 0.045
SWORD_TOUCH_THRESHOLD = 0.045
SWORD_TIP_ALIGN_Y = 0.05
SPREAD_THRESHOLD = 0.25
RESET_HOLD_TIME = 1.0
STILL_FRAMES_REQ = 5
MARGIN_X, MARGIN_Y = 80, 80
SCROLL_PINCH_THRESHOLD = 0.05
THUMB_OUT_X_DELTA = 0.055
THUMB_OUT_INDEX_DIST = 0.11

DEFAULT_SETTINGS = {
    "language": "zh",
    "target_hand": "Right",
    "sens_x": 1.8,
    "sens_y": 1.8,
    "smoothing": 5.0,
    "still_threshold": 0.002,
    "move_interval_ms": 16,
    "scroll_interval_ms": 35,
    "scroll_deadzone": 0.012,
    "scroll_scale": 1800,
    "scroll_step_limit": 90,
    "left_scroll_enter_frames": 3,
    "left_scroll_exit_frames": 4,
    "failsafe_enabled": True,
    "virtual_hand_mode": True,
    "theme": "light",
    "gesture_bindings": {
        "thumb_index": "left_drag",
        "thumb_middle": "right_click",
    },
}


def _deep_default_settings() -> dict[str, Any]:
    settings = DEFAULT_SETTINGS.copy()
    settings["gesture_bindings"] = DEFAULT_SETTINGS["gesture_bindings"].copy()
    return settings


def _normalize_settings(raw: dict[str, Any]) -> dict[str, Any]:
    settings = _deep_default_settings()
    if not isinstance(raw, dict):
        return settings

    for key, value in raw.items():
        if key == "gesture_bindings":
            continue
        if key in settings:
            settings[key] = value

    if isinstance(raw.get("gesture_bindings"), dict):
        settings["gesture_bindings"].update(raw["gesture_bindings"])

    if settings["target_hand"] not in ("Left", "Right"):
        settings["target_hand"] = DEFAULT_SETTINGS["target_hand"]
    if settings.get("language") not in ("zh", "en"):
        settings["language"] = DEFAULT_SETTINGS["language"]
    settings["failsafe_enabled"] = bool(settings.get("failsafe_enabled", True))
    return settings


def load_settings(settings_file: str | Path | None = None) -> dict[str, Any]:
    path = Path(settings_file) if settings_file else SETTINGS_FILE
    if not path.exists():
        return _deep_default_settings()

    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return _deep_default_settings()
    return _normalize_settings(loaded)


def save_settings(settings: dict[str, Any], settings_file: str | Path | None = None) -> None:
    path = Path(settings_file) if settings_file else SETTINGS_FILE
    normalized = _normalize_settings(settings)
    with path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=4, ensure_ascii=False)


def ensure_model_file(model_file: str | Path | None = None) -> bool:
    path = Path(model_file) if model_file else MODEL_FILE
    if path.exists() and path.stat().st_size > 0:
        return True

    try:
        import ssl
        context = ssl._create_unverified_context()
        with urlopen(MODEL_URL, timeout=20, context=context) as response:
            data = response.read()
        path.write_bytes(data)
    except (OSError, URLError, TimeoutError):
        return False
    return path.exists() and path.stat().st_size > 0


def get_dist_p(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def map_pointer_axis(val, v_min, v_max, sens):
    import numpy as np

    val = np.clip(val, v_min, v_max)
    norm = (val - v_min) / (v_max - v_min)
    return 0.5 + (norm - 0.5) * sens


def compute_scroll_step(tilt_abs: float, deadzone: float, scale: int, step_limit: int) -> int:
    effective = max(0.0, float(tilt_abs) - float(deadzone))
    raw = int(round(effective * int(scale)))
    return max(1, min(int(step_limit), raw))
