import math
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from PyQt6.QtCore import QThread, pyqtSignal
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.utils.config import (
    MARGIN_X,
    MARGIN_Y,
    MODEL_FILE,
    PINCH_THRESHOLD,
    PROCESS_H,
    PROCESS_W,
    RESET_HOLD_TIME,
    SPREAD_THRESHOLD,
    STILL_FRAMES_REQ,
    SWORD_TOUCH_THRESHOLD,
    compute_scroll_step,
    ensure_model_file,
    get_dist_p,
)

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False # 彻底禁用四角闪退保护

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

GESTURE_REFRESH_INTERVAL = 0.35


class HandTrackerThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    gesture_detected_signal = pyqtSignal(str, str)

    def __init__(self, settings):
        super().__init__()
        self.running = True
        self.settings = settings
        self.target_hand = self.settings["target_hand"]
        self.screen_w, self.screen_h = pyautogui.size()
        self.sens_x = self.settings["sens_x"]
        self.sens_y = self.settings["sens_y"]
        self.smoothing = self.settings["smoothing"]
        self.still_threshold = self.settings["still_threshold"]
        self.gesture_bindings = self.settings["gesture_bindings"].copy()
        self.move_interval_ms = self.settings["move_interval_ms"]
        self.scroll_interval_ms = self.settings["scroll_interval_ms"]
        self.scroll_deadzone = self.settings["scroll_deadzone"]
        self.scroll_scale = self.settings["scroll_scale"]
        self.scroll_step_limit = self.settings["scroll_step_limit"]
        self.left_scroll_enter_frames = self.settings["left_scroll_enter_frames"]
        self.left_scroll_exit_frames = self.settings["left_scroll_exit_frames"]
        # 强制禁用 Failsafe，忽略配置项以彻底解决闪退
        pyautogui.FAILSAFE = False

        self.last_gesture_state = {"left": "", "right": ""}
        self.last_gesture_emit_time = {"left": 0.0, "right": 0.0}

    def _emit_gesture(self, side, name):
        now = time.time()
        if (
            name != self.last_gesture_state[side]
            or now - self.last_gesture_emit_time[side] >= GESTURE_REFRESH_INTERVAL
        ):
            self.gesture_detected_signal.emit(side, name)
            self.last_gesture_state[side] = name
            self.last_gesture_emit_time[side] = now

    def set_target_hand(self, hand_type):
        self.target_hand = hand_type

    def set_sens(self, sens_x, sens_y):
        self.sens_x = sens_x
        self.sens_y = sens_y

    def set_smoothing(self, smoothing):
        self.smoothing = smoothing

    def set_still_threshold(self, threshold):
        self.still_threshold = threshold

    def set_gesture_binding(self, gesture, action):
        self.gesture_bindings[gesture] = action

    def set_move_interval_ms(self, interval_ms):
        self.move_interval_ms = interval_ms

    def set_scroll_profile(self, interval_ms, deadzone, scale, step_limit, enter_frames, exit_frames):
        self.scroll_interval_ms = interval_ms
        self.scroll_deadzone = deadzone
        self.scroll_scale = scale
        self.scroll_step_limit = step_limit
        self.left_scroll_enter_frames = enter_frames
        self.left_scroll_exit_frames = exit_frames

    def set_failsafe_enabled(self, enabled):
        pyautogui.FAILSAFE = bool(enabled)

    def _draw_hand(self, frame, landmarks, frame_w, frame_h, color, radius, thickness):
        if not landmarks:
            return
        for start, end in HAND_CONNECTIONS:
            p1 = landmarks[start]
            p2 = landmarks[end]
            cv2.line(
                frame,
                (int(p1.x * frame_w), int(p1.y * frame_h)),
                (int(p2.x * frame_w), int(p2.y * frame_h)),
                color,
                thickness,
                cv2.LINE_AA,
            )
        for landmark in landmarks:
            cv2.circle(
                frame,
                (int(landmark.x * frame_w), int(landmark.y * frame_h)),
                radius,
                color,
                -1,
            )

    def _handle_bound_action(self, action, is_left_pressed, last_right_click):
        gesture_name = ""
        if action == "left_drag":
            if not is_left_pressed:
                pyautogui.mouseDown()
                is_left_pressed = True
            gesture_name = "Left drag"
        elif action == "right_click":
            if time.time() - last_right_click > 0.8:
                pyautogui.rightClick()
                last_right_click = time.time()
            gesture_name = "Right click"
        return is_left_pressed, last_right_click, gesture_name

    def run(self):
        if not ensure_model_file(MODEL_FILE):
            self._emit_gesture("right", "Model unavailable")
            return

        base_options = python.BaseOptions(model_asset_path=str(MODEL_FILE))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            detector.close()
            self._emit_gesture("right", "Camera unavailable")
            return

        smooth_x, smooth_y = self.screen_w // 2, self.screen_h // 2
        pos_history = []
        is_cursor_locked = False
        reset_start_time = 0.0
        fist_window_end = 0.0
        last_right_click = 0.0
        is_left_pressed = False
        move_cooldown = 0
        scroll_enter_count = 0
        scroll_exit_count = 0
        scroll_active = False
        last_move_time = 0.0
        last_scroll_time = 0.0

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            now = time.time()

            small = cv2.resize(frame, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_NEAREST)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
            )
            results = detector.detect_for_video(mp_image, int(now * 1000))

            active_lms, helper_lms = None, None
            if results.hand_landmarks and results.handedness:
                for idx, hand_info in enumerate(results.handedness):
                    actual_hand = "Left" if hand_info[0].category_name == "Right" else "Right"
                    if actual_hand == self.target_hand:
                        active_lms = results.hand_landmarks[idx]
                    else:
                        helper_lms = results.hand_landmarks[idx]

            if active_lms:
                self._draw_hand(frame, active_lms, frame_w, frame_h, (30, 220, 255), 4, 2)
            if helper_lms:
                self._draw_hand(frame, helper_lms, frame_w, frame_h, (80, 180, 80), 3, 1)

            right_status = ""
            left_status = ""

            if active_lms:
                hand = active_lms
                wrist = hand[0]
                thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = hand[4], hand[8], hand[12], hand[16], hand[20]

                curled_fingers = sum(
                    1
                    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]
                    if hand[tip_idx].y > hand[pip_idx].y
                )
                is_fist = curled_fingers >= 3 and get_dist_p(thumb_tip, index_tip) < 0.1

                if self.target_hand == "Right":
                    is_back_facing = thumb_tip.x > hand[17].x
                else:
                    is_back_facing = thumb_tip.x < hand[17].x

                tips_to_wrist = [get_dist_p(wrist, point) for point in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]]
                is_spread = all(distance > SPREAD_THRESHOLD for distance in tips_to_wrist) and is_back_facing
                is_sword = get_dist_p(index_tip, middle_tip) < SWORD_TOUCH_THRESHOLD
                is_pinch_left = get_dist_p(thumb_tip, index_tip) < PINCH_THRESHOLD
                is_pinch_right = get_dist_p(thumb_tip, middle_tip) < PINCH_THRESHOLD

                if now < fist_window_end and is_fist:
                    pyautogui.hotkey("win", "d")
                    fist_window_end = 0.0
                    right_status = "Show desktop"

                if is_spread:
                    if reset_start_time == 0.0:
                        reset_start_time = now
                    elapsed = now - reset_start_time
                    progress = min(elapsed / RESET_HOLD_TIME, 1.0)
                    cv2.ellipse(
                        frame,
                        (int(wrist.x * frame_w), int(wrist.y * frame_h)),
                        (28, 28),
                        0,
                        0,
                        int(progress * 360),
                        (70, 210, 120),
                        3,
                    )
                    right_status = "Center reset"
                    if elapsed >= RESET_HOLD_TIME:
                        pyautogui.moveTo(self.screen_w // 2, self.screen_h // 2, _pause=False)
                        smooth_x, smooth_y = self.screen_w // 2, self.screen_h // 2
                        pos_history.clear()
                        is_cursor_locked = False
                        fist_window_end = now + 1.5
                        reset_start_time = -1.0
                else:
                    if reset_start_time == -1.0:
                        if not all(distance > SPREAD_THRESHOLD for distance in tips_to_wrist):
                            reset_start_time = 0.0
                    else:
                        reset_start_time = 0.0

                    if is_sword:
                        move_cooldown = 15
                        pos_history.append((index_tip.x, index_tip.y))
                        if len(pos_history) > STILL_FRAMES_REQ:
                            pos_history.pop(0)
                        if len(pos_history) == STILL_FRAMES_REQ:
                            xs = [point[0] for point in pos_history]
                            ys = [point[1] for point in pos_history]
                            spread = max(max(xs) - min(xs), max(ys) - min(ys))
                            if spread < self.still_threshold:
                                is_cursor_locked = True
                            elif spread > self.still_threshold * 4.0:
                                is_cursor_locked = False

                        right_status = "Move locked" if is_cursor_locked else "Moving"
                        if not is_cursor_locked:
                            rx1, ry1 = MARGIN_X, MARGIN_Y
                            rx2, ry2 = frame_w - MARGIN_X, frame_h - MARGIN_Y
                            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 220, 80), 1)
                            tx = np.interp(index_tip.x * frame_w, (rx1, rx2), (0, self.screen_w))
                            ty = np.interp(index_tip.y * frame_h, (ry1, ry2), (0, self.screen_h))
                            cx, cy = self.screen_w / 2, self.screen_h / 2
                            tx = cx + (tx - cx) * self.sens_x
                            ty = cy + (ty - cy) * self.sens_y
                            smooth_x += (tx - smooth_x) / self.smoothing
                            smooth_y += (ty - smooth_y) / self.smoothing
                            if (now - last_move_time) * 1000 >= self.move_interval_ms:
                                final_x = int(np.clip(smooth_x, 0, self.screen_w - 1))
                                final_y = int(np.clip(smooth_y, 0, self.screen_h - 1))
                                if not (math.isnan(final_x) or math.isnan(final_y)):
                                    pyautogui.moveTo(final_x, final_y, _pause=False)
                                    last_move_time = now
                    else:
                        pos_history.clear()
                        is_cursor_locked = False
                        if move_cooldown > 0:
                            move_cooldown -= 1
                        else:
                            active_action = None
                            if is_pinch_left:
                                active_action = self.gesture_bindings.get("thumb_index", "left_drag")
                            elif is_pinch_right:
                                active_action = self.gesture_bindings.get("thumb_middle", "right_click")

                            if active_action:
                                if active_action != "left_drag" and is_left_pressed:
                                    pyautogui.mouseUp()
                                    is_left_pressed = False
                                is_left_pressed, last_right_click, right_status = self._handle_bound_action(
                                    active_action,
                                    is_left_pressed,
                                    last_right_click,
                                )
                            else:
                                if is_left_pressed:
                                    pyautogui.mouseUp()
                                    is_left_pressed = False
                                right_status = "Waiting"

                cv2.circle(frame, (int(index_tip.x * frame_w), int(index_tip.y * frame_h)), 9, (40, 40, 220), -1)
            else:
                pos_history.clear()
                is_cursor_locked = False
                reset_start_time = 0.0
                if is_left_pressed:
                    pyautogui.mouseUp()
                    is_left_pressed = False

            if helper_lms:
                helper_index_tip = helper_lms[8]
                helper_middle_tip = helper_lms[12]
                helper_index_mcp = helper_lms[5]
                helper_middle_mcp = helper_lms[9]

                is_left_sword = get_dist_p(helper_index_tip, helper_middle_tip) < SWORD_TOUCH_THRESHOLD

                if is_left_sword:
                    scroll_enter_count += 1
                    scroll_exit_count = 0
                else:
                    scroll_exit_count += 1
                    if scroll_exit_count >= self.left_scroll_exit_frames:
                        scroll_enter_count = 0
                        scroll_active = False

                if scroll_enter_count >= self.left_scroll_enter_frames:
                    scroll_active = True

                if scroll_active and is_left_sword:
                    avg_tip_y = (helper_index_tip.y + helper_middle_tip.y) / 2.0
                    avg_base_y = (helper_index_mcp.y + helper_middle_mcp.y) / 2.0
                    tilt = avg_base_y - avg_tip_y
                    tilt_abs = abs(tilt)

                    left_status = "Scrolling"
                    if tilt_abs > self.scroll_deadzone and (now - last_scroll_time) * 1000 >= self.scroll_interval_ms:
                        step = compute_scroll_step(
                            tilt_abs=tilt_abs,
                            deadzone=self.scroll_deadzone,
                            scale=self.scroll_scale,
                            step_limit=self.scroll_step_limit,
                        )
                        scroll_delta = step if tilt > 0 else -step
                        pyautogui.scroll(scroll_delta)
                        last_scroll_time = now
                elif is_left_sword:
                    left_status = "Scroll armed"
                else:
                    left_status = "Waiting"
            else:
                scroll_enter_count = 0
                scroll_exit_count = 0
                scroll_active = False

            if right_status:
                self._emit_gesture("right", right_status)
            if left_status:
                self._emit_gesture("left", left_status)

            self.change_pixmap_signal.emit(frame)

        if is_left_pressed:
            pyautogui.mouseUp()
        cap.release()
        detector.close()
