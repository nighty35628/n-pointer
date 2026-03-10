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
        self.virtual_hand_mode = self.settings.get("virtual_hand_mode", True)
        # 强制禁用 Failsafe，忽略配置项以彻底解决闪退
        pyautogui.FAILSAFE = False

        self.last_gesture_state = {"left": "", "right": ""}
        self.last_gesture_emit_time = {"left": 0.0, "right": 0.0}

        # EMA 平滑存储
        self._prev_lms_active = None
        self._prev_lms_helper = None

        # 加载水墨背景
        self.bg_image = cv2.imread("assets/bg.png")
        if self.bg_image is None:
            # 如果加载失败，创建一个简单的白色背景
            self.bg_image = np.full((PROCESS_H, PROCESS_W, 3), 255, dtype=np.uint8)
        else:
            self.bg_image = cv2.resize(self.bg_image, (PROCESS_W, PROCESS_H))

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

    def _draw_ink_hand(self, frame, landmarks, frame_w, frame_h, side="active"):
        """
        Soft Ink Style Hand Renderer (Unified Shape)
        目标：使手部看起来像一个统一的软水墨形状，指关节不可见，边缘柔柔且略微模糊。
        """
        if not landmarks:
            return

        # --- STEP 1: Landmark Smoothing (EMA) ---
        curr_pts = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in landmarks])
        prev_attr = "_prev_lms_" + side
        prev_pts = getattr(self, prev_attr, None)
        
        if prev_pts is None:
            smoothed_pts = curr_pts
        else:
            smoothed_pts = 0.7 * curr_pts + 0.3 * prev_pts
        setattr(self, prev_attr, smoothed_pts)
        
        # --- STEP 2: Create an empty ink mask ---
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

        # --- STEP 3 & 4: Hand Size Calculation & Palm Base ---
        # 优化缩放逻辑：改用手掌区域（0,1,2,5,9,13,17）的整体半径作为缩放基准
        # 相比单一的纵向长度，由于包含了手掌所有核心点到中心的平均投影距离，
        # 在手指指向摄像头（即单点缩短）时，整体手掌的投影面积变化更平稳。
        palm_ref_pts = smoothed_pts[[0, 1, 2, 5, 9, 13, 17]]
        palm_center = np.mean(palm_ref_pts, axis=0)
        avg_palm_radius = np.mean(np.linalg.norm(palm_ref_pts - palm_center, axis=1))
        
        # 将 avg_palm_radius 标准化为比例系数（实验值：平均半径 50 像素左右为标准 1.0）
        hand_scale = avg_palm_radius / 55.0 
        
        # 限制缩放范围，防止极近或极远时失效，但允许它变得很小（从 0.2 开始）
        hand_scale = max(0.2, min(hand_scale, 2.5))

        # 包含手腕点 (0) 和手掌关键点，确保侧面时手腕处有足够的填充
        palm_indices = [0, 1, 2, 5, 9, 13, 17]
        palm_pts = smoothed_pts[palm_indices].astype(np.int32)
        hull = cv2.convexHull(palm_pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # 增强手掌根部：使用更丰满的椭圆，模拟圆角矩形的视觉感，并增加纵向延伸
        wrist_pt_raw = smoothed_pts[0]
        # 计算 2 到 17 的向量作为横向参考
        v_h = smoothed_pts[17] - smoothed_pts[2]
        v_h_unit = v_h / (np.linalg.norm(v_h) + 1e-6)
        
        # 在手腕位置绘制一个更丰满的椭圆
        angle = np.degrees(np.arctan2(v_h_unit[1], v_h_unit[0]))
        # 增加纵向（平行于手指方向）的半径从 15 增加到 22，横轴微增至 32，使边缘看起来更宽阔且不那么尖锐
        axes = (int(32 * hand_scale), int(22 * hand_scale)) 
        cv2.ellipse(mask, (int(wrist_pt_raw[0]), int(wrist_pt_raw[1])), axes, angle, 0, 360, 255, -1)
        
        # 为了让椭圆更像圆角矩形（边缘更平），我们在中心位置额外补一个小的填充块
        cv2.polylines(mask, [hull], True, 255, thickness=max(1, int(12 * hand_scale)))

        # --- STEP 5 & 6: Finger shapes (Soft brush path) ---
        # 指缝补偿逻辑：在相邻手指根部之间绘制圆块，防止背面检测导致的指缝过深、手指过长
        # 补偿点：食指-中指 (5, 9), 中指-无名指 (9, 13), 无名指-小指 (13, 17)
        webbing_pairs = [(5, 9), (9, 13), (13, 17)]
        for idx1, idx2 in webbing_pairs:
            p1, p2 = smoothed_pts[idx1], smoothed_pts[idx2]
            # 取两根手指根部的中点
            web_pt = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            # 绘制一个略向上偏移（向手指方向）的填充圆，
            # 这里的偏移系数 0.1 可以调节指缝的高度感
            cv2.circle(mask, (int(web_pt[0]), int(web_pt[1])), int(10 * hand_scale), 255, -1)

        # 恢复手掌厚度为之前较饱满的状态
        cv2.polylines(mask, [hull], True, 255, thickness=max(1, int(12 * hand_scale)))

        # 优化手指形状：前后直径更一致，使其更像均匀的圆柱体，避免关节处过粗 (糖葫芦感)
        # 将粗细进一步上调至 90% 左右（比之前的 80% 更饱满一点）
        # 每个手指的基准比例（从根部到指尖）进行微调：
        # 对于指向摄像头的手势，为了防止视觉过长，略微缩减指尖部分的延伸感
        finger_configs = [
            ([1, 2, 3, 4], 16 * hand_scale, 14 * hand_scale),   # 大拇指
            ([5, 6, 7, 8], 14 * hand_scale, 12 * hand_scale),   # 食指
            ([9, 10, 11, 12], 14 * hand_scale, 12 * hand_scale),# 中指
            ([13, 14, 15, 16], 13 * hand_scale, 11 * hand_scale),# 无名指
            ([17, 18, 19, 20], 12 * hand_scale, 10 * hand_scale) # 小指
        ]

        for indices, start_w, end_w in finger_configs:
            # 特殊处理指向摄像头时坐标重叠导致的视觉过长
            # 如果点与点之间像素距离过近，减少 linspace 绘制点数或强制收缩
            for i in range(len(indices) - 1):
                p1, p2 = smoothed_pts[indices[i]], smoothed_pts[indices[i+1]]
                seg_dist = np.linalg.norm(p1 - p2)
                # 当手指指向摄像头时，seg_dist 会非常小（透视收缩）
                # 这里我们增加一个简单的过滤，如果距离过近且处于指尖部分，不再强行画圆，防止堆叠导致“变长”
                num_steps = max(2, int(min(12, seg_dist / (2 * hand_scale))))
                
                for t in np.linspace(0, 1, num_steps):
                    x, y = int(p1[0]*(1-t)+p2[0]*t), int(p1[1]*(1-t)+p2[1]*t)
                    r = int(start_w*(1-t)+end_w*t)
                    # 确保半径最小为 1，且随距离缩小
                    cv2.circle(mask, (x, y), max(1, int(r)), 255, -1)

        # --- STEP 7: Render without blur ---
        # 移除所有高斯模糊逻辑，直接使用 mask
        mask_final = mask

        # --- STEP 8 & 9: Render asymmetric cross-border outline (Hollow effect) ---
        # 非对称描边：向内缩进约 2px，向外扩张约 1px
        # 这种“内多外少”的布局能让手指保持苗条，同时外边缘提供足够的定位感
        in_px = max(1, int(2 * hand_scale))
        out_px = max(1, int(1 * hand_scale))
        kernel = np.ones((3, 3), np.uint8)
        
        mask_dilated = cv2.dilate(mask_final, kernel, iterations=out_px)
        mask_eroded = cv2.erode(mask_final, kernel, iterations=in_px)
        
        # 描边 = 扩张后的外壳 - 缩进后的内核
        mixed_outline_mask = (mask_dilated.astype(np.float32) - mask_eroded.astype(np.float32)) / 255.0

        f_img = frame.astype(np.float32)
        
        # 描边透明度 (0.6)，主体内部保持中空
        outline_opacity = 0.6
        blend_alpha = mixed_outline_mask * outline_opacity

        alpha_3 = cv2.merge([blend_alpha, blend_alpha, blend_alpha])
        # 渲染这种非对称轮廓，视觉重心会略微向内收缩，解决偏粗的观感
        frame[:] = (f_img * (1.0 - alpha_3)).astype(np.uint8)

        # 移除二次模糊逻辑，保持绝对清晰度
        pass

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

            # 渲染逻辑
            if self.virtual_hand_mode:
                # 使用虚拟背景和水墨手 (新的分层渲染方案)
                display_frame = self.bg_image.copy()
                if active_lms:
                    self._draw_ink_hand(display_frame, active_lms, PROCESS_W, PROCESS_H, side="active")
                if helper_lms:
                    # 辅助手也使用同样的渲染逻辑
                    self._draw_ink_hand(display_frame, helper_lms, PROCESS_W, PROCESS_H, side="helper")
            else:
                # 传统预览模式
                display_frame = frame
                if active_lms:
                    self._draw_hand(display_frame, active_lms, frame_w, frame_h, (30, 220, 255), 4, 2)
                if helper_lms:
                    self._draw_hand(display_frame, helper_lms, frame_w, frame_h, (80, 180, 80), 3, 1)

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
                        display_frame,
                        (int(wrist.x * (PROCESS_W if self.virtual_hand_mode else frame_w)), 
                         int(wrist.y * (PROCESS_H if self.virtual_hand_mode else frame_h))),
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

            self.change_pixmap_signal.emit(display_frame)

        if is_left_pressed:
            pyautogui.mouseUp()
        cap.release()
        detector.close()
