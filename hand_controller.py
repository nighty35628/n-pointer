import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
import os
import urllib.request
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QSlider, QHBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
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
MARGIN_X, MARGIN_Y = 80, 80 
STILL_THRESHOLD = 0.005 
STILL_FRAMES_REQ = 5
PINCH_THRESHOLD = 0.045 
SWORD_TOUCH_THRESHOLD = 0.045 
SPREAD_THRESHOLD = 0.25 
RESET_HOLD_TIME = 1.0

MODEL_FILE = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

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

class HandTrackerThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True
        self.target_hand = "Right" 
        self.screen_w, self.screen_h = pyautogui.size()
        self.sens_x = 1.8 
        self.sens_y = 1.8
        self.smoothing = 3.0
        self.still_threshold = 0.002
        self.fist_window_end = 0 # Non-blocking window for Win+D
        
    def set_target_hand(self, hand_type):
        self.target_hand = hand_type 

    def set_sens(self, sx, sy):
        self.sens_x = sx
        self.sens_y = sy

    def set_smoothing(self, val):
        self.smoothing = val

    def set_still_threshold(self, val):
        self.still_threshold = val

    def run(self):
        download_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(0)
        
        smooth_x, smooth_y = self.screen_w // 2, self.screen_h // 2
        pos_history = []
        is_cursor_locked = False
        reset_start_time = 0
        prev_pos_yi = 0
        last_right_click = 0
        is_left_pressed = False
        move_cooldown = 0 

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            small = cv2.resize(frame, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_NEAREST)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            results = detector.detect_for_video(mp_image, int(time.time() * 1000))

            active_landmarks = None
            if results.hand_landmarks and results.handedness:
                for idx, hand_info in enumerate(results.handedness):
                    # Correcting Hand Swap: MediaPipe's "Left" is physical Right in mirror view
                    mp_label = hand_info[0].category_name
                    actual_hand = "Left" if mp_label == "Right" else "Right"
                    if actual_hand == self.target_hand:
                        active_landmarks = results.hand_landmarks[idx]
                        break
            
            if active_landmarks:
                hand = active_landmarks
                wrist = hand[0]
                thumb_tip, index_tip, middle_tip, ring_tip = hand[4], hand[8], hand[12], hand[16]
                
                dist_sword = get_dist_p(index_tip, middle_tip)
                is_sword = dist_sword < SWORD_TOUCH_THRESHOLD
                
                # Finger positions for Fist Detection: Tip higher Y than PIP (curled down)
                # More robust than simple distance to wrist
                curled_fingers = 0
                for tip_idx, pip_idx in [(8,6), (12,10), (16,14), (20,18)]:
                    if hand[tip_idx].y > hand[pip_idx].y: curled_fingers += 1
                is_fist = (curled_fingers >= 3) and (get_dist_p(thumb_tip, index_tip) < 0.1)

                # TRIGGER: SHOW DESKTOP (Win+D)
                # Only works for 1.5 seconds after a Reset finishes
                if time.time() < self.fist_window_end and is_fist:
                    pyautogui.hotkey('win', 'd')
                    self.fist_window_end = 0 # Prevent multiple triggers
                
                # Hand Orientation: Corrected for hand-swap
                if self.target_hand == "Right":
                    is_back_facing = hand[4].x > hand[17].x 
                else:
                    is_back_facing = hand[4].x < hand[17].x 
                
                tips_to_wrist = [get_dist_p(wrist, pt) for pt in [thumb_tip, index_tip, middle_tip, ring_tip, hand[20]]]
                is_spread = all(d > SPREAD_THRESHOLD for d in tips_to_wrist) and is_back_facing
                
                is_pinch_left = get_dist_p(thumb_tip, index_tip) < PINCH_THRESHOLD
                is_pinch_right = get_dist_p(thumb_tip, middle_tip) < PINCH_THRESHOLD
                is_pinch_scroll = get_dist_p(thumb_tip, ring_tip) < PINCH_THRESHOLD

                if is_spread:
                    if reset_start_time == 0: 
                        reset_start_time = time.time()
                    
                    if reset_start_time > 0:
                        elapsed = time.time() - reset_start_time
                        cv2.ellipse(frame, (int(wrist.x*w), int(wrist.y*h)), (30, 30), 0, 0, (elapsed/RESET_HOLD_TIME)*360, (0, 255, 0), 3)
                        if elapsed >= RESET_HOLD_TIME:
                            pyautogui.moveTo(self.screen_w // 2, self.screen_h // 2)
                            smooth_x, smooth_y = self.screen_w // 2, self.screen_h // 2
                            pos_history.clear(); is_cursor_locked = False
                            
                            # ENABLE THE FIST WINDOW (Non-blocking)
                            self.fist_window_end = time.time() + 1.5
                            reset_start_time = -1 
                    color = (0, 255, 0)
                else:
                    # Fix: Ensure logic resets properly if we were in the sentinel state
                    if reset_start_time == -1:
                        # Only clear the sentinel if we've actually let go of the spread gesture
                        # We use the distance tips-to-wrist to see if they are still spread
                        if not all(d > SPREAD_THRESHOLD for d in tips_to_wrist):
                            reset_start_time = 0
                    else:
                        reset_start_time = 0 
                    
                    # MOVEMENT MODE (Sword) - Higher Priority
                    if is_sword:
                        move_cooldown = 15 # Set cooldown (frames) to block clicks
                        pos_history.append((index_tip.x, index_tip.y))
                        if len(pos_history) > STILL_FRAMES_REQ: pos_history.pop(0)
                        if len(pos_history) == STILL_FRAMES_REQ:
                            xs, ys = [p[0] for p in pos_history], [p[1] for p in pos_history]
                            spread = max(max(xs)-min(xs), max(ys)-min(ys))
                            if spread < self.still_threshold: is_cursor_locked = True
                            elif spread > (self.still_threshold * 4.0): is_cursor_locked = False
                        
                        if not is_cursor_locked:
                            rx1, ry1 = MARGIN_X, MARGIN_Y
                            rx2, ry2 = w-MARGIN_X, h-MARGIN_Y
                            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
                            tx = np.interp(index_tip.x*w, (rx1, rx2), (0, self.screen_w))
                            ty = np.interp(index_tip.y*h, (ry1, ry2), (0, self.screen_h))
                            
                            # Center-based relative scaling for sensitivity
                            cx, cy = self.screen_w / 2, self.screen_h / 2
                            tx = cx + (tx - cx) * self.sens_x
                            ty = cy + (ty - cy) * self.sens_y

                            dx, dy = tx - smooth_x, ty - smooth_y
                            
                            smooth_x += dx / self.smoothing
                            smooth_y += dy / self.smoothing
                            
                            # Sanitize values before moving cursor
                            try:
                                final_x = int(np.clip(smooth_x, 0, self.screen_w - 1))
                                final_y = int(np.clip(smooth_y, 0, self.screen_h - 1))
                                if not (math.isnan(final_x) or math.isnan(final_y)):
                                    pyautogui.moveTo(final_x, final_y, _pause=False)
                            except: pass # Prevent crash on invalid move
                        color = (0, 215, 255)
                    else:
                        # ACTION MODE (Clicks & Scroll) - Only if NOT in move cooldown
                        pos_history.clear(); is_cursor_locked = False
                        if move_cooldown > 0:
                            move_cooldown -= 1 # Decrement wait time
                            color = (130, 0, 75)
                        else:
                            if is_pinch_left:
                                if not is_left_pressed: pyautogui.mouseDown(); is_left_pressed = True
                                color = (255, 0, 0)
                            else:
                                if is_left_pressed: pyautogui.mouseUp(); is_left_pressed = False
                                
                            if is_pinch_right:
                                if time.time() - last_right_click > 0.8:
                                    pyautogui.rightClick()
                                    last_right_click = time.time()
                                color = (0, 0, 255)
                            elif is_pinch_scroll:
                                try:
                                    scrl_dy = (index_tip.y - prev_pos_yi) * 3500
                                    if abs(scrl_dy) > 10:
                                        # Limit scroll delta to prevent extreme jerky movement or crashes
                                        safe_scroll = int(np.clip(-scrl_dy, -150, 150))
                                        pyautogui.scroll(safe_scroll)
                                except: pass
                                color = (255, 255, 0)
                            elif not is_pinch_left:
                                color = (130, 0, 75)

                cv2.circle(frame, (int(index_tip.x*w), int(index_tip.y*h)), 10, color, -1)
                prev_pos_yi = index_tip.y

            self.change_pixmap_signal.emit(frame)

        cap.release()
        detector.close()

class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("N-Pointer (All-in-One)")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        
        self.btn = QPushButton("Switch to LEFT Hand")
        self.btn.setFixedHeight(40)
        self.btn.clicked.connect(self.toggle_hand)
        self.layout.addWidget(self.btn)
        
        self.tracker = HandTrackerThread()
        self.tracker.change_pixmap_signal.connect(self.update_image)

        # Sensitivity Sliders (Hidden per user request, default values set)
        sens_container = QWidget()
        sens_layout = QVBoxLayout(sens_container)
        
        # Horizontal Sens
        h_box = QHBoxLayout()
        self.h_label = QLabel("H-Sens: 1.8")
        self.h_slider = QSlider(Qt.Orientation.Horizontal)
        self.h_slider.setRange(5, 30) 
        self.h_slider.setValue(18) # 1.8
        self.h_slider.valueChanged.connect(self.update_sens)
        h_box.addWidget(QLabel("H:"))
        h_box.addWidget(self.h_slider)
        h_box.addWidget(self.h_label)
        sens_layout.addLayout(h_box)

        # Vertical Sens
        v_box = QHBoxLayout()
        self.v_label = QLabel("V-Sens: 1.8")
        self.v_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_slider.setRange(5, 30)
        self.v_slider.setValue(18) # 1.8
        self.v_slider.valueChanged.connect(self.update_sens)
        v_box.addWidget(QLabel("V:"))
        v_box.addWidget(self.v_slider)
        v_box.addWidget(self.v_label)
        sens_layout.addLayout(v_box)

        # Smoothing Sens
        s_box = QHBoxLayout()
        self.s_label = QLabel("Smooth: 3.0")
        self.s_slider = QSlider(Qt.Orientation.Horizontal)
        self.s_slider.setRange(1, 20) 
        self.s_slider.setValue(3) # 3.0
        self.s_slider.valueChanged.connect(self.update_sens)
        s_box.addWidget(QLabel("S:"))
        s_box.addWidget(self.s_slider)
        s_box.addWidget(self.s_label)
        sens_layout.addLayout(s_box)

        # Still Threshold (Lock Range)
        t_box = QHBoxLayout()
        self.t_label = QLabel("Lock: 0.002")
        self.t_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_slider.setRange(1, 40) 
        self.t_slider.setValue(2) # 0.002
        self.t_slider.valueChanged.connect(self.update_sens)
        t_box.addWidget(QLabel("L:"))
        t_box.addWidget(self.t_slider)
        t_box.addWidget(self.t_label)
        sens_layout.addLayout(t_box)

        sens_container.hide() # Hide everything as requested
        self.layout.addWidget(sens_container)
        
        self.tracker.start()
        
    def update_sens(self):
        sx = self.h_slider.value() / 10.0
        sy = self.v_slider.value() / 10.0
        sm = float(self.s_slider.value())
        th = self.t_slider.value() / 1000.0
        self.h_label.setText(f"H-Sens: {sx:.1f}")
        self.v_label.setText(f"V-Sens: {sy:.1f}")
        self.s_label.setText(f"Smooth: {sm:.1f}")
        self.t_label.setText(f"Lock: {th:.3f}")
        self.tracker.set_sens(sx, sy)
        self.tracker.set_smoothing(sm)
        self.tracker.set_still_threshold(th)

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def toggle_hand(self):
        if self.tracker.target_hand == "Right":
            self.tracker.set_target_hand("Left")
            self.btn.setText("Switch to RIGHT Hand")
        else:
            self.tracker.set_target_hand("Right")
            self.btn.setText("Switch to LEFT Hand")

    def closeEvent(self, event):
        self.tracker.running = False
        self.tracker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ControlWindow()
    window.show()
    sys.exit(app.exec())
