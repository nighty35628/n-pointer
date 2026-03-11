import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
import os
import urllib.request
import sys
import json
import random
from collections import deque
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QSlider, QHBoxLayout, QStackedWidget, QComboBox, QFormLayout, QGroupBox, QFrame, QSizePolicy, QScrollArea, QBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PyQt6.QtGui import QImage, QPixmap
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- PERFORMANCE & SAFETY CONFIG ---
pyautogui.PAUSE = 0  
pyautogui.FAILSAFE = False

# Performance scaling
PROCESS_W = 320 
PROCESS_H = 240

# Values & Thresholds
EDGE_PAD_X = 0.04
EDGE_PAD_Y = 0.06
EDGE_CURVE = 0.72
MIN_DYNAMIC_SMOOTHING = 1.15
STILL_THRESHOLD = 0.005 
STILL_FRAMES_REQ = 5
PINCH_THRESHOLD = 0.045 
SWORD_TOUCH_THRESHOLD = 0.065 # Increased for easier sword detection (was 0.050)
SWORD_TIP_ALIGN_Y = 0.12
HAND_LOSS_HOLD_TIME = 0.35
HAND_REACQUIRE_DISTANCE = 0.18
THUMB_CANCEL_X_DELTA = 0.055
THUMB_CANCEL_INDEX_DIST = 0.11
THUMB_CANCEL_FRAMES = 3
SPREAD_THRESHOLD = 0.25 
RESET_HOLD_TIME = 1.0

MODEL_FILE = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

DEFAULT_SETTINGS = {
    "target_hand": "Right",
    "sens_x": 1.8,
    "sens_y": 1.8,
    "smoothing": 2.0,
    "still_threshold": 0.001,
    "gesture_bindings": {
        "thumb_index": "left_drag",
        "thumb_middle": "right_click",
    },
}

GESTURE_ACTION_OPTIONS = [
    ("左键", "left_drag"),
    ("右键", "right_click"),
    ("无", "none"),
]

GESTURE_DESCRIPTIONS = [
    "右手剑指：进入并保持移动状态",
    "右手伸出大拇指：暂停移动（收回后恢复）",
    "左手食指中指并拢并向上/向下指：控制滚轮",
    "可配置手势 1：右手大拇指 + 食指捏合",
    "可配置手势 2：右手大拇指 + 中指捏合",
]



def load_settings():
    settings = {
        **DEFAULT_SETTINGS,
        "gesture_bindings": DEFAULT_SETTINGS["gesture_bindings"].copy(),
    }
    if not os.path.exists(SETTINGS_FILE):
        return settings

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return settings

    settings.update({k: v for k, v in loaded.items() if k != "gesture_bindings"})
    if isinstance(loaded.get("gesture_bindings"), dict):
        settings["gesture_bindings"].update(loaded["gesture_bindings"])
    return settings


def save_settings(settings):
    payload = {
        "target_hand": settings["target_hand"],
        "sens_x": settings["sens_x"],
        "sens_y": settings["sens_y"],
        "smoothing": settings["smoothing"],
        "still_threshold": settings["still_threshold"],
        "gesture_bindings": settings["gesture_bindings"],
    }
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def download_model():
    if not os.path.exists(MODEL_FILE):
        try:
            import ssl
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(MODEL_URL, context=context) as response:
                with open(MODEL_FILE, "wb") as f:
                    f.write(response.read())
        except: sys.exit(1)

def get_dist_p(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def map_pointer_axis(value, pad, curve, sensitivity):
    usable = np.clip((value - pad) / (1.0 - 2.0 * pad), 0.0, 1.0)
    centered = (usable - 0.5) * 2.0
    curved = math.copysign(abs(centered) ** curve, centered)
    scaled = np.clip(curved * sensitivity, -1.0, 1.0)
    return (scaled + 1.0) * 0.5

class HandTrackerThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    gesture_detected_signal = pyqtSignal(str) 

    def __init__(self, settings=None):
        super().__init__()
        self.running = True
        self.settings = settings or load_settings()
        self.target_hand = self.settings["target_hand"]
        self.screen_w, self.screen_h = pyautogui.size()
        self.sens_x = self.settings["sens_x"]
        self.sens_y = self.settings["sens_y"]
        self.smoothing = self.settings["smoothing"]
        self.still_threshold = self.settings["still_threshold"]
        self.gesture_bindings = self.settings["gesture_bindings"].copy()
        
        # 水墨竹林状态初始化
        self.paper_color = (230, 240, 245) # 宣纸色 (BGR)
        self.ink_black = (20, 20, 30)      # 水墨深色
        self.bamboos = []
        for _ in range(6):
            self.bamboos.append({
                'x': random.randint(20, PROCESS_W - 20),
                'width': random.uniform(1.0, 3.0),
                'phase': random.uniform(0, 2 * math.pi),
                'h_factor': random.uniform(0.8, 1.1) # 随机高度系数
            })
        
        self.leaves = []
        for _ in range(15):
            self.leaves.append(self._create_leaf())

        # 指尖拖尾 (只保留食指和中指)
        self.trails = [deque(maxlen=15) for _ in range(2)]
        self.fingertip_indices = [8, 12] # 食指和中指
        
        self.fist_window_end = 0 
        self.last_gesture_sent = ""
        self.gesture_clear_time = 0

    def _create_leaf(self):
        """创建一个具有物理属性的水墨竹叶"""
        return {
            'x0': random.randint(-50, PROCESS_W + 50),
            'y': random.uniform(-100, -10),
            'speed': random.uniform(0.5, 1.5),
            'amp': random.uniform(10, 30),      # 摆动幅度
            'freq': random.uniform(1.0, 2.5),   # 摆动频率
            'rot_speed': random.uniform(-2, 2), # 旋转速度
            'angle': random.uniform(0, 360),
            'size': random.uniform(0.6, 1.2),
            't': 0
        }

    def _draw_ink_bamboo(self, img, bx, base_w, phase):
        """垂直顶点偏移摇摆竹子绘制"""
        h_total = PROCESS_H
        segments = 10
        points = []
        t = time.time()
        
        # 计算顶点位置：dx = A * sin(t) * (h/H)^2
        for i in range(segments + 1):
            h_curr = (i / segments) * h_total
            # 顶部 (i=segments) 摆动最大, 底部 (i=0) 不动
            sway_amp = 15 * math.sin(t * 0.7 + phase)
            dx = sway_amp * ((h_curr / h_total) ** 2)
            
            # y 轴从下往上 (PROCESS_H -> 0)
            py = int(h_total - h_curr)
            px = int(bx + dx)
            points.append((px, py))

        # 绘制竹竿段
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            # 颜色随深度略微变化，模拟水墨浓淡
            color = (130, 140, 130) # 中等灰色
            cv2.line(img, p1, p2, color, int(base_w), cv2.LINE_AA)
            
            # 绘制竹节 (节点)
            if i > 0:
                node_w = int(base_w + 2)
                cv2.line(img, (p1[0] - node_w, p1[1]), (p1[0] + node_w, p1[1]), (100, 110, 100), 1, cv2.LINE_AA)

    def _draw_ink_leaf(self, img, leaf):
        """绘制“个”字形水墨竹叶"""
        # 物理模拟：x = x0 + A * sin(f * t)
        leaf['t'] += 0.02
        leaf['y'] += leaf['speed']
        leaf['angle'] += leaf['rot_speed']
        
        curr_x = leaf['x0'] + leaf['amp'] * math.sin(leaf['freq'] * leaf['t'])
        center = (int(curr_x), int(leaf['y']))
        
        # “个”字形由3-4个笔触组成
        # 笔触相对于中心的偏移（旋转前）
        strokes = [
            [(0, 0), (0, 15)],      # 主轴
            [(0, 3), (-8, 10)],     # 左侧撇
            [(0, 3), (8, 10)]       # 右侧捺
        ]
        
        color = (110, 120, 110)
        angle_rad = math.radians(leaf['angle'])
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for start, end in strokes:
            # 旋转与缩放计算
            def transform(p):
                x, y = p[0] * leaf['size'], p[1] * leaf['size']
                rx = x * cos_a - y * sin_a
                ry = x * sin_a + y * cos_a
                return (int(center[0] + rx), int(center[1] + ry))

            p1 = transform(start)
            p2 = transform(end)
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

    def _render_background(self, img):
        """渲染完整的水墨背景"""
        # 背景底色
        img[:] = self.paper_color
        
        # 绘制竹林
        for b in self.bamboos:
            self._draw_ink_bamboo(img, b['x'], b['width'], b['phase'])

        # 绘制落叶
        for i in range(len(self.leaves)):
            leaf = self.leaves[i]
            self._draw_ink_leaf(img, leaf)
            
            # 边界重置
            if leaf['y'] > PROCESS_H + 20:
                self.leaves[i] = self._create_leaf()

    def _draw_ink_line(self, img, p1, p2, color, thickness):
        """绘制具有毛笔质感的线条（带随机粗细和晕染）"""
        # 主线条
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        
        # 模拟毛笔分叉和晕染效果
        if thickness > 1:
            # 外部淡晕
            cv2.line(img, p1, p2, (40, 40, 50), thickness + 3, cv2.LINE_AA)
            # 内部干笔笔触（随机偏移的小线条）
            for _ in range(2):
                off = random.randint(-1, 1)
                p1_off = (p1[0] + off, p1[1] + off)
                p2_off = (p2[0] + off, p2[1] + off)
                cv2.line(img, p1_off, p2_off, (10, 10, 15), 1, cv2.LINE_AA)

    def _draw_virtual_hand(self, img, landmarks, is_active=True):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
        ]
        h, w = img.shape[:2]
        ink_color = self.ink_black if is_active else (180, 180, 180)
        
        # 1. 绘制笔触骨架
        for start_idx, end_idx in connections:
            pt1, pt2 = landmarks[start_idx], landmarks[end_idx]
            p1 = (int(pt1.x * w), int(pt1.y * h))
            p2 = (int(pt2.x * w), int(pt2.y * h))
            
            # 墨迹感：线宽随深度变化
            z_avg = (pt1.z + pt2.z) / 2
            thickness = max(1, int(4 * (1.1 - (z_avg + 0.2))))
            # 使用改进的毛笔线条绘制
            if is_active:
                self._draw_ink_line(img, p1, p2, ink_color, thickness)
            else:
                cv2.line(img, p1, p2, ink_color, thickness, cv2.LINE_AA)

        # 2. 绘制重要关节 (墨点)
        for i, pt in enumerate(landmarks):
            p = (int(pt.x * w), int(pt.y * h))
            if i in [4, 8, 12, 16, 20]: # 保持所有手指末端墨点，即使没有拖尾
                cv2.circle(img, p, 5, ink_color, -1, cv2.LINE_AA)
                cv2.circle(img, p, 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, p, 2, (80, 80, 80), -1, cv2.LINE_AA)
        
        if is_active:
            self._draw_accessories(img, landmarks)

    def _draw_accessories(self, img, landmarks):
        h, w = img.shape[:2]
        # 在手腕处绘制朱红色印章标记
        wrist = landmarks[0]
        p = (int(wrist.x * w), int(wrist.y * h))
        # 红色印章矩形
        cv2.rectangle(img, (p[0]-12, p[1]-12), (p[0]+12, p[1]+12), (40, 40, 200), -1, cv2.LINE_AA)
        cv2.rectangle(img, (p[0]-10, p[1]-10), (p[0]+10, p[1]+10), (255, 255, 255), 1, cv2.LINE_AA)
        # 印章伪文字线条
        cv2.line(img, (p[0]-5, p[1]-5), (p[0]+5, p[1]-5), (255, 255, 255), 1)
        cv2.line(img, (p[0]-5, p[1]+5), (p[0]+5, p[1]+5), (255, 255, 255), 1)

    def _emit_gesture(self, gesture_name):
        if gesture_name != self.last_gesture_sent or time.time() > self.gesture_clear_time:
            self.gesture_detected_signal.emit(gesture_name)
            self.last_gesture_sent = gesture_name
            self.gesture_clear_time = time.time() + 1.2 

    def set_target_hand(self, h): self.target_hand = h
    def set_sens(self, sx, sy): self.sens_x, self.sens_y = sx, sy
    def set_smoothing(self, v): self.smoothing = v
    def set_still_threshold(self, v): self.still_threshold = v
    def set_gesture_binding(self, g, a): self.gesture_bindings[g] = a
    def release_mouse_buttons(self):
        try: pyautogui.mouseUp(button="left")
        except: pass

    def run(self):
        download_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, running_mode=vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(0)
        
        smooth_x, smooth_y = self.screen_w // 2, self.screen_h // 2
        pos_history = []
        is_cursor_locked = False
        reset_start_time = 0
        is_left_pressed = False
        is_move_mode = False
        move_mode_hold_until = 0.0
        last_active_index_point = None
        thumb_cancel_frames = 0
        last_right_click = 0
        move_cooldown = 0

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            # 背景渲染：宣纸底色 + 竹林
            virtual_frame = np.full((PROCESS_H, PROCESS_W, 3), self.paper_color, dtype=np.uint8)
            self._render_background(virtual_frame)
            
            small = cv2.resize(frame, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_NEAREST)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            results = detector.detect_for_video(mp_image, int(time.time() * 1000))

            active_lms, other_lms, active_idx = None, None, None
            hand_candidates = []
            if results.hand_landmarks and results.handedness:
                for idx, hi in enumerate(results.handedness):
                    ah = "Left" if hi[0].category_name == "Right" else "Right"
                    hand_candidates.append((idx, ah, results.hand_landmarks[idx]))

                for idx, ah, lms in hand_candidates:
                    if ah == self.target_hand: active_idx, active_lms = idx, lms; break
                
                if active_lms is None and is_move_mode and last_active_index_point:
                    bd = HAND_REACQUIRE_DISTANCE
                    for idx, _, lms in hand_candidates:
                        d = math.dist((lms[8].x, lms[8].y), last_active_index_point)
                        if d < bd: bd, active_idx, active_lms = d, idx, lms

                for idx, _, lms in hand_candidates:
                    if idx != active_idx: other_lms = lms; break

            if other_lms:
                self._draw_virtual_hand(virtual_frame, other_lms, is_active=False)
                # 滚轮逻辑
                if get_dist_p(other_lms[8], other_lms[12]) < SWORD_TOUCH_THRESHOLD:
                    avg_b = (other_lms[8].y - other_lms[6].y + other_lms[12].y - other_lms[10].y) / 2
                    if abs(avg_b) > 0.02:
                        pyautogui.scroll(int(-avg_b * 500))
                        self._emit_gesture("滚动控制")

            if active_lms:
                # 渲染指尖拖尾 (只处理食指 8 和中指 12)
                for i, tip_idx in enumerate(self.fingertip_indices):
                    pt = active_lms[tip_idx]
                    self.trails[i].append((int(pt.x * PROCESS_W), int(pt.y * PROCESS_H)))
                    for j in range(1, len(self.trails[i])):
                        alpha = j / len(self.trails[i])
                        # 拖尾颜色深浅变化
                        color = (int(30 + 100*(1-alpha)), int(30 + 100*(1-alpha)), int(40 + 100*(1-alpha)))
                        cv2.line(virtual_frame, self.trails[i][j-1], self.trails[i][j], color, max(1, int(4*alpha)), cv2.LINE_AA)

                self._draw_virtual_hand(virtual_frame, active_lms, is_active=True)
                
                # 手势行为控制
                hand = active_lms
                wrist = hand[0]
                thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = hand[4], hand[8], hand[12], hand[16], hand[20]
                curled = sum(1 for t, p in [(8,6), (12,10), (16,14), (20,18)] if hand[t].y > hand[p].y)
                
                # A. 状态判断：手背张开复位
                if self.target_hand == "Right":
                    is_back_facing = thumb_tip.x > hand[17].x 
                else:
                    is_back_facing = thumb_tip.x < hand[17].x
                tips_to_wrist = [get_dist_p(wrist, pt) for pt in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]]
                is_spread = all(d > 0.15 for d in tips_to_wrist) and is_back_facing

                # B. 复位计时与握拳监测
                if is_spread:
                    if is_move_mode: # 开始转圈时，立即停止当前的移动或点击状态
                        is_move_mode = False
                        if is_left_pressed:
                            pyautogui.mouseUp()
                            is_left_pressed = False
                        self._emit_gesture("屏蔽移动")
                            
                    if reset_start_time == 0: reset_start_time = time.time()
                    elapsed = time.time() - reset_start_time
                    progress = min(elapsed / 1.5, 1.0)
                    
                    # 绘制复位进度圆环
                    cv2.ellipse(virtual_frame, (int(wrist.x*PROCESS_W), int(wrist.y*PROCESS_H)), (20, 20), 0, 0, int(progress*360), (40, 40, 200), 2)
                    
                    if progress >= 1.0:
                        # 圈满：执行复位并标记为待触发状态
                        if reset_start_time != -1: 
                            pyautogui.moveTo(self.screen_w // 2, self.screen_h // 2)
                            smooth_x, smooth_y = self.screen_w // 2, self.screen_h // 2
                            is_move_mode = False
                            self._emit_gesture("中心复位")
                            reset_start_time = -1 # 保持 -1 状态，直到触发握拳
                else:
                    # 如果不是张开状态，检查是否需要触发握拳返回桌面
                    if reset_start_time == -1: # 只有在复位圈转满之后
                        is_fist = (curled >= 4) and (get_dist_p(thumb_tip, index_tip) < 0.1)
                        if is_fist:
                            pyautogui.hotkey('win', 'd')
                            reset_start_time = 0 # 触发后重置
                            self._emit_gesture("显示桌面")
                        else:
                            # 剑指触发判定：如果在待命状态下做出了剑指手势，也相当于取消待命回到移动模式
                            is_sword_gesture = (get_dist_p(index_tip, middle_tip) < SWORD_TOUCH_THRESHOLD and 
                                              hand[8].y < hand[6].y and hand[10].y < hand[14].y and curled <= 3)
                            if is_sword_gesture:
                                reset_start_time = 0
                                self._emit_gesture("取消复位")
                    else:
                        reset_start_time = 0

                # 3. 剑指平移逻辑 (仅在非复位计时期间执行，且非复位待命状态)
                if reset_start_time == 0:
                    thumb_out = (thumb_tip.x < hand[2].x - THUMB_CANCEL_X_DELTA) if self.target_hand == "Right" else (thumb_tip.x > hand[2].x + THUMB_CANCEL_X_DELTA)
                    is_thumb_ext = thumb_out and get_dist_p(thumb_tip, index_tip) > THUMB_CANCEL_INDEX_DIST
                    
                    is_sword = (get_dist_p(index_tip, middle_tip) < SWORD_TOUCH_THRESHOLD and 
                               abs(index_tip.y - middle_tip.y) < SWORD_TIP_ALIGN_Y and 
                               hand[8].y < hand[6].y and hand[10].y < hand[14].y and curled <= 3 and not is_thumb_ext)

                    if is_sword:
                        is_move_mode, move_mode_hold_until = True, time.time() + HAND_LOSS_HOLD_TIME
                        self._emit_gesture("进入移动")
                    
                    if is_thumb_ext:
                        thumb_cancel_frames += 1
                    else: 
                        thumb_cancel_frames = 0

                    if is_move_mode and not is_thumb_ext:
                        last_active_index_point = (index_tip.x, index_tip.y)
                        move_mode_hold_until = time.time() + HAND_LOSS_HOLD_TIME
                        tx = map_pointer_axis(index_tip.x, EDGE_PAD_X, EDGE_CURVE, self.sens_x) * self.screen_w
                        ty = map_pointer_axis(index_tip.y, EDGE_PAD_Y, EDGE_CURVE, self.sens_y) * self.screen_h
                        smooth_x += (tx - smooth_x) / self.smoothing
                        smooth_y += (ty - smooth_y) / self.smoothing
                        pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)
                    elif is_move_mode and is_thumb_ext:
                        if thumb_cancel_frames >= THUMB_CANCEL_FRAMES:
                            self._emit_gesture("暂停移动")

                # 4. 点击控制
                p_l = get_dist_p(thumb_tip, index_tip) < PINCH_THRESHOLD
                p_r = get_dist_p(thumb_tip, middle_tip) < PINCH_THRESHOLD
                if (p_l and self.gesture_bindings["thumb_index"] == "left_drag") or \
                   (p_r and self.gesture_bindings["thumb_middle"] == "left_drag"):
                    if not is_left_pressed: pyautogui.mouseDown(); is_left_pressed = True; self._emit_gesture("左键按住")
                elif is_left_pressed: pyautogui.mouseUp(); is_left_pressed = False
            else:
                for t in self.trails: t.clear()
                if time.time() >= move_mode_hold_until: is_move_mode, last_active_index_point = False, None
                if is_left_pressed: pyautogui.mouseUp(); is_left_pressed = False

            self.change_pixmap_signal.emit(virtual_frame)

        cap.release(); detector.close()

class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("N-Pointer")
        self.resize(1020, 680)
        self.settings = load_settings()
        self.is_loading_ui = True
        self.sidebar_user_collapsed = False
        self.auto_sidebar_collapsed = False

        self.central_widget = QWidget()
        self.central_widget.setObjectName("appRoot")
        self.setCentralWidget(self.central_widget)
        self.root_layout = QHBoxLayout(self.central_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        # --- Sidebar Setup ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(210)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(12, 24, 12, 24)
        self.sidebar_layout.setSpacing(8)

        self.menu_btn = QPushButton("☰")
        self.menu_btn.setObjectName("menuButton")
        self.menu_btn.setFixedSize(42, 42)
        self.menu_btn.clicked.connect(self.toggle_sidebar)
        self.sidebar_layout.addWidget(self.menu_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.sidebar_layout.addSpacing(20)

        self.home_btn = QPushButton(" 实时控制")
        self.home_btn.setObjectName("navButton")
        self.home_btn.setCheckable(True)
        self.settings_btn = QPushButton(" 偏好设置")
        self.settings_btn.setObjectName("navButton")
        self.settings_btn.setCheckable(True)
        
        # Navigation Items with Emoji as Icons
        self.nav_items = [
            (self.home_btn, "🎮", " 实时控制"),
            (self.settings_btn, "⚙️", " 偏好设置"),
        ]
        
        for btn, icon, _ in self.nav_items:
            btn.clicked.connect(self._handle_nav_click)
            self.sidebar_layout.addWidget(btn)

        self.sidebar_layout.addStretch()

        # Moved Brand to Bottom
        self.brand_card = QFrame()
        self.brand_card.setObjectName("brandCard")
        self.brand_layout = QVBoxLayout(self.brand_card)
        self.brand_layout.setContentsMargins(12, 12, 12, 12)
        self.brand_layout.setSpacing(4)
        self.brand_title = QLabel("N-POINTER")
        self.brand_title.setObjectName("brandTitle")
        self.brand_subtitle = QLabel("V2.0 Core")
        self.brand_subtitle.setObjectName("brandSubtitle")
        self.brand_layout.addWidget(self.brand_title)
        self.brand_layout.addWidget(self.brand_subtitle)
        self.sidebar_layout.addWidget(self.brand_card)

        # --- Main Content Area ---
        self.content_area = QFrame()
        self.content_area.setObjectName("contentArea")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(0)

        self.pages = QStackedWidget()
        self.pages.setObjectName("pageStack")

        # Page 0: Home
        self.home_page = QWidget()
        self.home_layout = QVBoxLayout(self.home_page)
        self.home_layout.setContentsMargins(0, 0, 0, 0)
        self.home_layout.setSpacing(20)

        self.page_title = QLabel("实时控制")
        self.page_title.setObjectName("pageTitle")
        self.home_layout.addWidget(self.page_title)

        # Main Row for Preview and Gesture Cards
        self.home_main_row = QWidget()
        self.home_main_row_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, self.home_main_row)
        self.home_main_row_layout.setContentsMargins(0, 0, 0, 0)
        self.home_main_row_layout.setSpacing(20)

        # Preview Container (Shrunk to 1/4 area roughly by limiting width/height)
        self.preview_card = QFrame()
        self.preview_card.setObjectName("panelCard")
        self.preview_card_layout = QVBoxLayout(self.preview_card)
        self.preview_card_layout.setContentsMargins(10, 10, 10, 10)
        self.preview_card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center vertically inside card
        
        self.image_label = QLabel()
        self.image_label.setObjectName("previewSurface")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 360) 
        self.image_label.setMaximumSize(960, 720) 
        self.preview_card_layout.addWidget(self.image_label)
        
        # Gesture Info Container
        self.gesture_panel = QWidget()
        self.gesture_panel.setMinimumWidth(240) # Stable width
        self.gesture_panel_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, self.gesture_panel)
        self.gesture_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.gesture_panel_layout.setSpacing(16)
        
        # Card 1: Right Hand
        self.gesture_card_r = QFrame()
        self.gesture_card_r.setObjectName("gestureCard")
        self.gesture_card_r.setMinimumHeight(140) # Fixed height for stability
        self.layout_r = QVBoxLayout(self.gesture_card_r)
        self.layout_r.addWidget(QLabel("主控手状态").setObjectName("sectionTitle"))
        self.gesture_label_r = QLabel("等待识别")
        self.gesture_label_r.setObjectName("gestureDisplay")
        self.gesture_label_r.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout_r.addWidget(self.gesture_label_r)
        self.gesture_panel_layout.addWidget(self.gesture_card_r)

        # Card 2: Left Hand
        self.gesture_card_l = QFrame()
        self.gesture_card_l.setObjectName("gestureCard")
        self.gesture_card_l.setMinimumHeight(140)
        self.layout_l = QVBoxLayout(self.gesture_card_l)
        self.layout_l.addWidget(QLabel("辅助手状态").setObjectName("sectionTitle"))
        self.gesture_label_l = QLabel("空闲")
        self.gesture_label_l.setObjectName("gestureDisplay")
        self.gesture_label_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout_l.addWidget(self.gesture_label_l)
        self.gesture_panel_layout.addWidget(self.gesture_card_l)

        
        self.home_main_row_layout.addWidget(self.preview_card, 2)
        self.home_main_row_layout.addWidget(self.gesture_panel, 1)
        
        self.home_layout.addStretch(1) # Top stretch
        self.home_layout.addWidget(self.home_main_row)
        self.home_layout.addStretch(2) # Bottom stretch to keep it centered properly

        self.home_hint = QLabel("💡 提示：右手剑指进入移动态，大拇指外翘退出；左手二指并拢控制滚轮。")
        self.home_hint.setObjectName("infoText")
        self.home_layout.addWidget(self.home_hint)

        # Page 1: Settings
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setObjectName("settingsScroll")
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.settings_page = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_page)
        self.settings_layout.setContentsMargins(0, 0, 40, 20)
        self.settings_layout.setSpacing(24)

        self.settings_title_label = QLabel("偏好设置")
        self.settings_title_label.setObjectName("pageTitle")
        self.settings_layout.addWidget(self.settings_title_label)

        self.settings_grid = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.settings_grid.setSpacing(20)

        self.settings_left_col = QVBoxLayout()
        self.settings_left_col.setSpacing(20)
        self.settings_right_col = QVBoxLayout()
        self.settings_right_col.setSpacing(20)

        # Settings Panels
        self.general_panel = QFrame()
        self.general_panel.setObjectName("panelCard")
        self.general_form = QFormLayout(self.general_panel)
        self.general_form.setContentsMargins(20, 20, 20, 20)
        self.general_form.setSpacing(18)
        self.general_form.addRow(QLabel("常规设置").setObjectName("sectionTitle"))

        self.hand_combo = QComboBox()
        self.hand_combo.addItem("右手识别", "Right")
        self.hand_combo.addItem("左手识别", "Left")
        self.general_form.addRow("主控手", self.hand_combo)

        self.h_label = QLabel()
        self.h_slider = QSlider(Qt.Orientation.Horizontal)
        self.h_slider.setRange(5, 30)
        self.general_form.addRow("水平灵敏度", self._wrap_slider(self.h_slider, self.h_label))

        self.v_label = QLabel()
        self.v_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_slider.setRange(5, 30)
        self.general_form.addRow("垂直灵敏度", self._wrap_slider(self.v_slider, self.v_label))

        self.s_label = QLabel()
        self.s_slider = QSlider(Qt.Orientation.Horizontal)
        self.s_slider.setRange(5, 50) # Expanded range for percentage 
        self.general_form.addRow("平滑度", self._wrap_slider(self.s_slider, self.s_label))

        self.t_label = QLabel()
        self.t_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_slider.setRange(1, 100) # 0.0001 to 0.0100
        self.general_form.addRow("锁定阈值", self._wrap_slider(self.t_slider, self.t_label))

        self.settings_left_col.addWidget(self.general_panel)


        self.binding_panel = QFrame()
        self.binding_panel.setObjectName("panelCard")
        self.binding_form = QFormLayout(self.binding_panel)
        self.binding_form.setContentsMargins(20, 20, 20, 20)
        self.binding_form.setSpacing(18)
        self.binding_form.addRow(QLabel("按键映射").setObjectName("sectionTitle"))
        
        self.thumb_index_combo = QComboBox()
        self.thumb_middle_combo = QComboBox()
        for label, value in GESTURE_ACTION_OPTIONS:
            self.thumb_index_combo.addItem(label, value)
            self.thumb_middle_combo.addItem(label, value)
        self.binding_form.addRow("大拇指+食指捏合", self.thumb_index_combo)
        self.binding_form.addRow("大拇指+中指捏合", self.thumb_middle_combo)
        self.settings_left_col.addWidget(self.binding_panel)

        self.help_panel = QFrame()
        self.help_panel.setObjectName("panelCard")
        self.help_layout = QVBoxLayout(self.help_panel)
        self.help_layout.setContentsMargins(20, 20, 20, 20)
        self.help_layout.setSpacing(12)
        self.help_layout.addWidget(QLabel("手势说明").setObjectName("sectionTitle"))
        for desc in GESTURE_DESCRIPTIONS:
            hl = QLabel(desc)
            hl.setObjectName("gestureHint")
            hl.setWordWrap(True)
            self.help_layout.addWidget(hl)
        self.settings_right_col.addWidget(self.help_panel)

        self.settings_grid.addLayout(self.settings_left_col, 3)
        self.settings_grid.addLayout(self.settings_right_col, 2)
        self.settings_layout.addLayout(self.settings_grid)
        self.settings_layout.addStretch()
        self.settings_scroll.setWidget(self.settings_page)

        self.pages.addWidget(self.home_page)
        self.pages.addWidget(self.settings_scroll)
        self.content_layout.addWidget(self.pages)

        self.root_layout.addWidget(self.sidebar)
        self.root_layout.addWidget(self.content_area, 1)

        # Sidebar Animations
        self.sidebar_anim = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self.sidebar_anim.setDuration(300)
        self.sidebar_anim.setEasingCurve(QEasingCurve.Type.OutQuint)
        self.sidebar_anim_max = QPropertyAnimation(self.sidebar, b"maximumWidth")
        self.sidebar_anim_max.setDuration(300)
        self.sidebar_anim_max.setEasingCurve(QEasingCurve.Type.OutQuint)

        self.tracker = HandTrackerThread(self.settings)
        self.tracker.change_pixmap_signal.connect(self.update_image)
        self.tracker.gesture_detected_signal.connect(self.on_gesture_detected)

        self.hand_combo.currentIndexChanged.connect(self.apply_preferences)
        self.h_slider.valueChanged.connect(self.apply_preferences)
        self.v_slider.valueChanged.connect(self.apply_preferences)
        self.s_slider.valueChanged.connect(self.apply_preferences)
        self.t_slider.valueChanged.connect(self.apply_preferences)
        self.thumb_index_combo.currentIndexChanged.connect(self.apply_preferences)
        self.thumb_middle_combo.currentIndexChanged.connect(self.apply_preferences)

        self.load_settings_to_ui()
        self.set_page(0)
        self.apply_modern_style()
        self.update_sidebar_ui(animated=False)
        self.is_loading_ui = False
        self.apply_preferences()
        
        self.tracker.start()

    def _handle_nav_click(self):
        sender = self.sender()
        index = 0 if sender == self.home_btn else 1
        self.set_page(index)

    def _wrap_slider(self, slider, value_label):
        container = QWidget()
        container.setObjectName("sliderWrap")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)
        layout.addWidget(slider, 1)
        value_label.setMinimumWidth(50)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setObjectName("valuePill")
        layout.addWidget(value_label)
        return container

    def on_gesture_detected(self, name):
        if name in ["滚动控制"]:
            target = self.gesture_label_l
        else:
            target = self.gesture_label_r
            
        target.setText(name)
        target.setStyleSheet("color: #fa2d48; font-weight: bold; font-size: 18px;")
        QTimer.singleShot(800, lambda: target.setStyleSheet(""))

    def apply_modern_style(self):
        self.setStyleSheet("""
            QWidget#appRoot {
                background: #f8f8fa;
                color: #1d1d1f;
                font-family: "Segoe UI", "PingFang SC", sans-serif;
            }
            QFrame#sidebar {
                background: #f2f2f7;
                border_right: 1px solid #e1e1e6;
            }
            QFrame#contentArea {
                background: transparent;
            }
            QPushButton#menuButton {
                background: transparent;
                border: none;
                font-size: 20px;
                color: #424245;
                border-radius: 8px;
            }
            QPushButton#menuButton:hover {
                background: #e1e1e6;
            }
            QPushButton#navButton {
                background: transparent;
                border: none;
                border-radius: 10px;
                padding: 12px 14px;
                text-align: left;
                font-size: 15px;
                font-weight: 500;
                color: #424245;
            }
            QPushButton#navButton:hover {
                background: #e1e1e6;
            }
            QPushButton#navButton:checked {
                background: #fa2d48;
                color: #ffffff;
            }
            QFrame#brandCard {
                background: transparent;
                border-top: 1px solid #e1e1e6;
                padding-top: 15px;
            }
            QLabel#brandTitle {
                color: #1d1d1f;
                font-size: 14px;
                font-weight: 800;
                letter-spacing: 1.5px;
            }
            QLabel#brandSubtitle {
                color: #86868b;
                font-size: 11px;
            }
            QLabel#pageTitle {
                font-size: 28px;
                font-weight: 700;
                color: #1d1d1f;
                margin-bottom: 20px;
            }
            QFrame#panelCard, QFrame#gestureCard {
                background: #ffffff;
                border: 1px solid #eaeaea;
                border-radius: 20px;
            }
            QLabel#sectionTitle {
                font-size: 16px;
                font-weight: 600;
                color: #1d1d1f;
                margin-bottom: 5px;
            }
            QLabel#gestureDisplay {
                font-size: 22px;
                font-weight: 400;
                color: #fa2d48;
                margin: 10px 0;
            }
            QLabel#previewSurface {
                background: #fdfdfb; /* 宣纸纹理底色 */
                border-radius: 20px;
                border: 4px solid #ffffff;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }
            QLabel#infoText, QLabel#gestureHint {
                color: #86868b;
                font-size: 13px;
                line-height: 1.5;
            }
            QLabel#gestureHint {
                padding: 10px;
                background: #fdfdfd;
                border: 1px solid #efeff3;
                border-radius: 10px;
            }
            QLabel#valuePill {
                background: #ffffff;
                color: #fa2d48;
                border: 1px solid #e5e5e5;
                border-radius: 12px;
                padding: 6px 0;
                font-weight: 700;
                font-size: 12px;
            }
            #sliderWrap {
                margin: 4px 0;
                padding: 10px 0;
                background: transparent;
            }
            QComboBox {
                background: #ffffff;
                border: 1px solid #d2d2d7;
                border-radius: 10px;
                padding: 6px 12px;
            }
            QComboBox:hover {
                border-color: #fa2d48;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #86868b;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #e1e1e6;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #fa2d48;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 3px solid #ffffff;
                width: 22px;
                height: 22px;
                margin: -9px 0;
                border-radius: 11px;
            }
        """)



    def set_page(self, index):
        self.pages.setCurrentIndex(index)
        self.home_btn.setChecked(index == 0)
        self.settings_btn.setChecked(index == 1)

    def toggle_sidebar(self):
        self.sidebar_user_collapsed = not self.sidebar_user_collapsed
        self.update_sidebar_ui(animated=True)

    def update_sidebar_ui(self, animated=True):
        collapsed = self.sidebar_user_collapsed or self.auto_sidebar_collapsed
        target_width = 72 if collapsed else 210

        self.sidebar.setProperty("collapsed", "true" if collapsed else "false")
        self.sidebar.style().unpolish(self.sidebar)
        self.sidebar.style().polish(self.sidebar)

        self.brand_card.setVisible(not collapsed)
        for button, icon, full_text in self.nav_items:
            button.setText(icon if collapsed else icon + full_text)
            button.setToolTip(full_text if collapsed else "")
            if collapsed:
                button.setFixedSize(48, 48)
                button.setStyleSheet("text-align: center; font-size: 18px; padding: 0;")
            else:
                button.setFixedSize(186, 46)
                button.setStyleSheet("")

        if animated:
            self.sidebar_anim.stop()
            self.sidebar_anim_max.stop()
            self.sidebar_anim.setStartValue(self.sidebar.width())
            self.sidebar_anim.setEndValue(target_width)
            self.sidebar_anim_max.setStartValue(self.sidebar.width())
            self.sidebar_anim_max.setEndValue(target_width)
            self.sidebar_anim.start()
            self.sidebar_anim_max.start()
        else:
            self.sidebar.setFixedWidth(target_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_responsive_layout()

    def refresh_responsive_layout(self):
        width = self.width()
        is_narrow = width < 880 
        
        # Consistent auto-collapse & layout shift
        self.auto_sidebar_collapsed = width < 880 
        
        # Toggle Settings Grid
        new_dir = QBoxLayout.Direction.TopToBottom if is_narrow else QBoxLayout.Direction.LeftToRight
        if self.settings_grid.direction() != new_dir:
            # Add simple fade animation during layout shift
            self.pages.setGraphicsEffect(None)
            self.settings_grid.setDirection(new_dir)
            
        # Toggle Home Layout
        if self.home_main_row_layout.direction() != new_dir:
            self.home_main_row_layout.setDirection(new_dir)
            self.gesture_panel_layout.setDirection(
                QBoxLayout.Direction.LeftToRight if is_narrow else QBoxLayout.Direction.TopToBottom
            )
            
        self.update_sidebar_ui(animated=True)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap with Aspect Ratio Fill"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fill the label while keeping aspect ratio
        target_w = self.image_label.width()
        target_h = self.image_label.height()
        
        p = QPixmap.fromImage(qt_image)
        return p.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

        self.hand_combo.setCurrentIndex(0 if self.settings["target_hand"] == "Right" else 1)
        self.h_slider.setValue(int(self.settings["sens_x"] * 10))
        self.v_slider.setValue(int(self.settings["sens_y"] * 10))
        self.s_slider.setValue(int(self.settings["smoothing"]))
        self.t_slider.setValue(int(self.settings["still_threshold"] * 1000))
        self._set_combo_by_data(self.thumb_index_combo, self.settings["gesture_bindings"]["thumb_index"])
        self._set_combo_by_data(self.thumb_middle_combo, self.settings["gesture_bindings"]["thumb_middle"])
        self.refresh_setting_labels()

    def _set_combo_by_data(self, combo, value):
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def refresh_setting_labels(self):
        # Scaling inputs to percent based on max range
        p_hx = int((self.h_slider.value() / 30.0) * 100)
        p_vx = int((self.v_slider.value() / 30.0) * 100)
        p_sm = int((self.s_slider.value() / 50.0) * 100)
        p_th = int((self.t_slider.value() / 100.0) * 100)
        
        self.h_label.setText(f"{p_hx}%")
        self.v_label.setText(f"{p_vx}%")
        self.s_label.setText(f"{p_sm}%")
        self.t_label.setText(f"{p_th}%")

    def apply_preferences(self):
        self.refresh_setting_labels()
        if self.is_loading_ui:
            return
        
        self.settings["target_hand"] = self.hand_combo.currentData()
        self.settings["sens_x"] = self.h_slider.value() / 10.0
        self.settings["sens_y"] = self.v_slider.value() / 10.0
        self.settings["smoothing"] = float(self.s_slider.value())
        # Threshold: 1 -> 0.0001, 100 -> 0.0100
        self.settings["still_threshold"] = self.t_slider.value() / 10000.0
        
        self.settings["gesture_bindings"]["thumb_index"] = self.thumb_index_combo.currentData()
        self.settings["gesture_bindings"]["thumb_middle"] = self.thumb_middle_combo.currentData()
        
        save_settings(self.settings)

        # Update Thread live
        self.tracker.set_target_hand(self.settings["target_hand"])
        self.tracker.set_sens(self.settings["sens_x"], self.settings["sens_y"])
        self.tracker.set_smoothing(self.settings["smoothing"])
        self.tracker.set_still_threshold(self.settings["still_threshold"])
        self.tracker.set_gesture_binding("thumb_index", self.settings["gesture_bindings"]["thumb_index"])
        self.tracker.set_gesture_binding("thumb_middle", self.settings["gesture_bindings"]["thumb_middle"])

        self.settings["still_threshold"] = self.t_slider.value() / 1000.0
        self.settings["gesture_bindings"]["thumb_index"] = self.thumb_index_combo.currentData()
        self.settings["gesture_bindings"]["thumb_middle"] = self.thumb_middle_combo.currentData()
        
        save_settings(self.settings)
        self.tracker.set_target_hand(self.settings["target_hand"])
        self.tracker.set_sens(self.settings["sens_x"], self.settings["sens_y"])
        self.tracker.set_smoothing(self.settings["smoothing"])
        self.tracker.set_still_threshold(self.settings["still_threshold"])
        self.tracker.set_gesture_binding("thumb_index", self.settings["gesture_bindings"]["thumb_index"])
        self.tracker.set_gesture_binding("thumb_middle", self.settings["gesture_bindings"]["thumb_middle"])

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Calculate aspect ratio to avoid black bars
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        
        # Center crop or scale to fill without black bars
        # For simplicity, we scale to the smaller dimension's ratio and crop the rest
        # But standard Scale is usually fine if we use KeepAspectRatioByExpanding or just careful KeepAspectRatio
        p = convert_to_Qt_format.scaled(label_w, label_h, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        
        # Crop to exact label size
        p = p.copy((p.width() - label_w) // 2, (p.height() - label_h) // 2, label_w, label_h)
        
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.tracker.running = False
        self.tracker.wait()
        event.accept()

    def load_settings_to_ui(self):
        self.hand_combo.setCurrentIndex(0 if self.settings["target_hand"] == "Right" else 1)
        self.h_slider.setValue(int(self.settings["sens_x"] * 10))
        self.v_slider.setValue(int(self.settings["sens_y"] * 10))
        self.s_slider.setValue(int(self.settings["smoothing"]))
        # Convert back from 0.0001-0.0100 to 1-100 range
        self.t_slider.setValue(int(self.settings["still_threshold"] * 10000))
        self._set_combo_by_data(self.thumb_index_combo, self.settings["gesture_bindings"]["thumb_index"])
        self._set_combo_by_data(self.thumb_middle_combo, self.settings["gesture_bindings"]["thumb_middle"])
        self.refresh_setting_labels()


    def _set_combo_by_data(self, combo, value):
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ControlWindow()
    window.show()
    sys.exit(app.exec())
