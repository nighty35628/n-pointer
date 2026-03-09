import cv2

from PyQt6.QtCore import QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, Qt, QTimer, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QRegion, QIcon
from PyQt6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QGraphicsDropShadowEffect,
)

from src.core.tracker import HandTrackerThread
from src.utils.config import load_settings, save_settings

SIDEBAR_WIDE = 210
SIDEBAR_NARROW = 70
# 同步阈值：侧边栏收缩与布局切换共享此阈值
RESPONSIVE_THRESHOLD = 680 
MIN_W = 405
MIN_H = 500

I18N = {
    "zh": {
        "title": "N-Pointer",
        "version": "v1.0.2",
        "brand_sub": "手势鼠标控制器",
        "nav_home": "实时控制",
        "nav_settings": "偏好设置",
        "toggle_sidebar": "收起/展开侧栏",
        "hero": "欢迎使用 N-Pointer",
        "hero_sub": "简单几步，让手势控制更准确更流畅。",
        "status_main": "主控手状态",
        "status_helper": "辅助手状态",
        "waiting": "等待识别...",
        "guide": "当前说明",
        "hint_1": "主控手剑指用于控制鼠标移动。",
        "hint_2": "主控手五指张开保持可将光标复位到中心。",
        "hint_3": "拇指+食指、拇指+中指可映射动作。",
        "hint_4": "辅助手剑指用于滚动控制。",
        "general": "基础设置",
        "language": "语言",
        "primary": "主控手",
        "failsafe": "启用 PyAutoGUI 紧急保护",
        "h_sens": "水平灵敏度",
        "v_sens": "垂直灵敏度",
        "smooth": "平滑度",
        "lock": "锁定阈值",
        "binding": "动作映射",
        "thumb_index": "拇指 + 食指",
        "thumb_middle": "拇指 + 中指",
        "perf": "性能参数",
        "move_i": "移动间隔 (ms)",
        "scroll_i": "滚动间隔 (ms)",
        "scroll_d": "滚动死区",
        "scroll_s": "滚动倍率",
        "notes": "说明",
        "n1": "窗口变窄时会自动收起侧栏。",
        "n2": "卡片布局会随窗口尺寸自动重排。",
        "n3": "支持无边框圆角与深色模式切换。",
        "theme": "主题",
        "theme_light": "浅色模式",
        "theme_dark": "深色模式",
        "right": "右手",
        "left": "左手",
        "lang_zh": "中文",
        "lang_en": "English",
        "g_ld": "左键",
        "g_rc": "右键",
        "g_none": "无动作",
        "st_wait": "等待识别...",
        "st_ld": "左键控制",
        "st_rc": "右键点击",
        "st_model": "模型不可用",
        "st_cam": "摄像头不可用",
        "st_show": "显示桌面",
        "st_reset": "中心复位",
        "st_lock": "移动锁定",
        "st_move": "正在移动",
        "st_scroll": "滚动中",
        "st_scroll_arm": "滚动预备",
        "ctrl_start": "启动控制",
        "ctrl_stop": "停止控制",
    },
    "en": {
        "title": "N-Pointer",
        "brand_sub": "Hand-gesture mouse controller",
        "nav_home": "Live Control",
        "nav_settings": "Preferences",
        "toggle_sidebar": "Toggle sidebar",
        "hero": "Welcome to N-Pointer",
        "hero_sub": "A few steps to keep your gesture control stable and smooth.",
        "status_main": "Main hand status",
        "status_helper": "Helper hand status",
        "waiting": "Waiting...",
        "guide": "Current Guide",
        "hint_1": "Main hand sword pose controls cursor movement.",
        "hint_2": "Open-hand hold resets cursor to center.",
        "hint_3": "Thumb+index and thumb+middle actions are configurable.",
        "hint_4": "Helper-hand sword pose controls scrolling.",
        "general": "General",
        "language": "Language",
        "theme": "Theme",
        "primary": "Primary hand",
        "failsafe": "Enable PyAutoGUI failsafe",
        "h_sens": "Horizontal sensitivity",
        "v_sens": "Vertical sensitivity",
        "smooth": "Smoothing",
        "lock": "Lock threshold",
        "binding": "Gesture binding",
        "thumb_index": "Thumb + index",
        "thumb_middle": "Thumb + middle",
        "perf": "Performance",
        "move_i": "Move interval (ms)",
        "scroll_i": "Scroll interval (ms)",
        "scroll_d": "Scroll deadzone",
        "scroll_s": "Scroll scale",
        "notes": "Notes",
        "n1": "Sidebar auto-collapses when window is narrow.",
        "n2": "Cards switch to vertical layout on narrow widths.",
        "n3": "Transitions are animated for smoother changes.",
        "right": "Right hand",
        "left": "Left hand",
        "lang_zh": "Chinese",
        "lang_en": "English",
        "g_ld": "Left Click",
        "g_rc": "Right Click",
        "g_none": "None",
        "st_wait": "Waiting...",
        "st_ld": "Left Ctrl",
        "st_rc": "Right Click",
        "st_model": "Model unavailable",
        "st_cam": "Camera unavailable",
        "st_show": "Show desktop",
        "st_reset": "Center reset",
        "st_lock": "Move locked",
        "st_move": "Moving",
        "st_scroll": "Scrolling",
        "st_scroll_arm": "Scroll armed",
        "theme_light": "Light Mode",
        "theme_dark": "Dark Mode",
        "ctrl_start": "Start Control",
        "ctrl_stop": "Stop Control",
        "version": "v1.0.2",
    },
}

THEMES = {
    "light": {
        "sidebar_bg": "#f2f2f7",
        "content_bg": "#ffffff",
        "card_bg": "#f2f2f7",
        "text_primary": "#000000",
        "text_secondary": "#3c3c43",
        "accent": "#fa2d48",
        "border": "#d1d1d6",
        "hover": "#e5e5ea",
        "input_bg": "#ffffff",
        "preview_bg": "#000000",
    },
    "dark": {
        "sidebar_bg": "#1c1c1e",
        "content_bg": "#000000",
        "card_bg": "#1c1c1e",
        "text_primary": "#ffffff",
        "text_secondary": "#8e8e93",
        "accent": "#fa2d48",
        "border": "#38383a",
        "hover": "#2c2c2e",
        "input_bg": "#2c2c2e",
        "preview_bg": "#000000",
    }
}

STATUS_MAP = {
    "Left drag": "st_ld",
    "Right click": "st_rc",
    "Model unavailable": "st_model",
    "Camera unavailable": "st_cam",
    "Show desktop": "st_show",
    "Center reset": "st_reset",
    "Move locked": "st_lock",
    "Moving": "st_move",
    "Waiting": "st_wait",
    "Scrolling": "st_scroll",
    "Scroll armed": "st_scroll_arm",
}


class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.language = self.settings.get("language", "zh") if self.settings.get("language") in ("zh", "en") else "zh"
        self.theme_name = self.settings.get("theme", "light")
        self.sidebar_expanded = True
        self.compact_mode = False
        self.reflow_busy = False
        self.is_loading_ui = True

        self.setWindowTitle(self._t("title"))
        self.setMinimumSize(MIN_W, MIN_H)
        self.resize(1180, 780)
        self._init_ui()

        self.tracker = HandTrackerThread(self.settings)
        self.tracker.change_pixmap_signal.connect(self.update_image)
        self.tracker.gesture_detected_signal.connect(self.on_gesture_detected)

        self._connect_signals()
        self.load_settings_to_ui()
        self.set_page(0)
        self.apply_style()
        self.apply_language()
        self.is_loading_ui = False
        self.apply_preferences()
        self.tracker.start()

        # 默认不开启控制：初始设为 False，并不调用 tracker.start(如果你希望线程运行但不处理逻辑)
        # 或者更彻底：在这里不 start，直到点击按钮
        self.tracker.running = False 

        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(lambda: self._apply_responsive())
        self._apply_responsive(animate=False)

    def _t(self, key):
        return I18N[self.language][key]

    def _init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        self.root_layout = QHBoxLayout(root)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        # Main contain to help with rounding
        self.main_container = QFrame(objectName="mainContainer")
        self.main_container_layout = QHBoxLayout(self.main_container)
        self.main_container_layout.setContentsMargins(0, 0, 0, 0)
        self.main_container_layout.setSpacing(0)
        self.root_layout.addWidget(self.main_container)

        self.sidebar = QFrame(objectName="sidebar")
        self.sidebar.setMinimumWidth(SIDEBAR_WIDE)
        self.sidebar.setMaximumWidth(SIDEBAR_WIDE)
        s = QVBoxLayout(self.sidebar)
        s.setContentsMargins(12, 14, 12, 14)
        s.setSpacing(8)

        self.toggle_btn = QPushButton("≡", objectName="toggleButton")
        self.toggle_btn.setStyleSheet("font-weight: bold; font-size: 24px;")
        s.addWidget(self.toggle_btn)

        self.brand_box = QWidget()
        bb = QVBoxLayout(self.brand_box)
        bb.setContentsMargins(0, 0, 0, 0)
        bb.setSpacing(0)
        self.brand_title = QLabel("N-POINTER", objectName="brandTitle")
        self.brand_subtitle = QLabel(objectName="brandSubtitle")
        bb.addWidget(self.brand_title)
        bb.addWidget(self.brand_subtitle)
        s.addWidget(self.brand_box)
        self.brand_box.hide() # 初始在上方隐藏，稍后移动到下方
        
        s.addSpacing(16)

        self.home_btn = QPushButton(objectName="navButton")
        self.home_btn.setCheckable(True)
        self.settings_btn = QPushButton(objectName="navButton")
        self.settings_btn.setCheckable(True)
        s.addWidget(self.home_btn)
        s.addWidget(self.settings_btn)
        s.addStretch()

        # 品牌标识移动到左下角
        self.brand_container = QWidget()
        bc = QVBoxLayout(self.brand_container)
        bc.setContentsMargins(8, 0, 8, 4)
        bc.setSpacing(2)
        
        self.footer_title = QLabel("N-POINTER", objectName="brandTitle")
        self.footer_subtitle = QLabel(objectName="brandSubtitle")
        self.sidebar_version = QLabel("v1.0.2", objectName="sidebarInfoVersion")
        
        bc.addWidget(self.footer_title)
        bc.addWidget(self.footer_subtitle)
        bc.addWidget(self.sidebar_version)
        s.addWidget(self.brand_container)

        self.content = QFrame()
        self.content.setObjectName("contentArea")
        c = QVBoxLayout(self.content)
        c.setContentsMargins(48, 40, 48, 40)
        c.setSpacing(20)
        self.pages = QStackedWidget()
        self.pages.setObjectName("stackedPages")
        self.pages.setContentsMargins(0, 0, 0, 0)
        self._init_home_page()
        self._init_settings_page()
        c.addWidget(self.pages)

        self.main_container_layout.addWidget(self.sidebar)
        self.main_container_layout.addWidget(self.content, 1)

        self.gesture_timers = {"left": QTimer(self), "right": QTimer(self)}
        self.gesture_timers["left"].setSingleShot(True)
        self.gesture_timers["right"].setSingleShot(True)
        self.gesture_timers["left"].timeout.connect(lambda: self.gesture_label_l.setText(self._t("waiting")))
        self.gesture_timers["right"].timeout.connect(lambda: self.gesture_label_r.setText(self._t("waiting")))

    def _toggle_theme(self):
        self.theme_name = "dark" if self.theme_name == "light" else "light"
        self.settings["theme"] = self.theme_name
        save_settings(self.settings)
        self.apply_style()
        self.apply_language()
        self.update() # trigger repaint

    def _init_home_page(self):
        self.home_page = QWidget()
        l = QVBoxLayout(self.home_page)
        l.setSpacing(12)
        
        top_bar = QHBoxLayout()
        hero_box = QVBoxLayout()
        self.hero = QLabel(objectName="heroTitle")
        self.hero_sub = QLabel(objectName="heroSub")
        self.hero_sub.setWordWrap(True)
        hero_box.addWidget(self.hero)
        hero_box.addWidget(self.hero_sub)
        top_bar.addLayout(hero_box, 1)
        
        self.start_stop_btn = QPushButton(objectName="powerButton")
        self.start_stop_btn.setCheckable(True)
        self.start_stop_btn.setChecked(False) # 默认不开启
        self.start_stop_btn.setFixedSize(140, 50)
        top_bar.addWidget(self.start_stop_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        l.addLayout(top_bar)

        self.home_cards = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.home_cards.setSpacing(14)
        self.preview_card = QFrame(objectName="panelCard")
        pv = QVBoxLayout(self.preview_card)
        pv.setContentsMargins(1, 1, 1, 1) # 几乎不留边距，让预览撑满圆角卡片
        
        self.image_label = QLabel(objectName="previewSurface")
        self.image_label.setMinimumSize(400, 300) # 降低最小硬性限制，改用缩放
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pv.addWidget(self.image_label)

        self.status_panel = QWidget()
        self.status_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, self.status_panel)
        self.status_layout.setContentsMargins(0, 0, 0, 0)
        self.status_layout.setSpacing(14)
        self.status_title_main, self.gesture_label_r = self._create_status_card(self.status_layout)
        self.status_title_helper, self.gesture_label_l = self._create_status_card(self.status_layout)
        self.status_layout.addStretch()

        self.home_cards.addWidget(self.preview_card, 3)
        self.home_cards.addWidget(self.status_panel, 2)
        l.addLayout(self.home_cards)

        self.hint_card = QFrame(objectName="panelCard")
        hl = QVBoxLayout(self.hint_card)
        self.hint_title = QLabel(objectName="sectionTitle")
        hl.addWidget(self.hint_title)
        self.hint_labels = []
        for _ in range(4):
            lb = QLabel(objectName="hintText")
            lb.setWordWrap(True)
            self.hint_labels.append(lb)
            hl.addWidget(lb)
        l.addWidget(self.hint_card)
        self.pages.addWidget(self.home_page)

    def _create_status_card(self, parent):
        card = QFrame(objectName="panelCard")
        l = QVBoxLayout(card)
        t = QLabel(objectName="sectionTitle")
        v = QLabel(objectName="gestureDisplay")
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(t)
        l.addWidget(v)
        parent.addWidget(card)
        return t, v

    def _init_settings_page(self):
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setObjectName("settingsScroll")
        self.settings_page = QWidget()
        self.settings_page.setObjectName("settingsPage")
        self.settings_cards = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.settings_cards.setContentsMargins(0, 0, 0, 0)
        self.settings_cards.setSpacing(16)

        self.left_col = QWidget()
        left = QVBoxLayout(self.left_col)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(16)
        self.right_col = QWidget()
        right = QVBoxLayout(self.right_col)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(16)

        self.general_card = QFrame(objectName="panelCard")
        gf = QFormLayout(self.general_card)
        
        # 语言单选组
        self.lang_group = QButtonGroup(self)
        self.lang_zh_rb = QRadioButton("中文"); self.lang_en_rb = QRadioButton("English")
        self.lang_group.addButton(self.lang_zh_rb, 0); self.lang_group.addButton(self.lang_en_rb, 1)
        lang_layout = QHBoxLayout(); lang_layout.addWidget(self.lang_zh_rb); lang_layout.addWidget(self.lang_en_rb); lang_layout.addStretch()
        
        # 主题单选组
        self.theme_group = QButtonGroup(self)
        self.theme_light_rb = QRadioButton(); self.theme_dark_rb = QRadioButton()
        self.theme_group.addButton(self.theme_light_rb, 0); self.theme_group.addButton(self.theme_dark_rb, 1)
        theme_layout = QHBoxLayout(); theme_layout.addWidget(self.theme_light_rb); theme_layout.addWidget(self.theme_dark_rb); theme_layout.addStretch()
        
        # 主控手单选组
        self.hand_group = QButtonGroup(self)
        self.hand_right_rb = QRadioButton(); self.hand_left_rb = QRadioButton()
        self.hand_group.addButton(self.hand_right_rb, 0); self.hand_group.addButton(self.hand_left_rb, 1)
        hand_layout = QHBoxLayout(); hand_layout.addWidget(self.hand_right_rb); hand_layout.addWidget(self.hand_left_rb); hand_layout.addStretch()

        self.h_slider = QSlider(Qt.Orientation.Horizontal); self.h_slider.setRange(5, 30)
        self.v_slider = QSlider(Qt.Orientation.Horizontal); self.v_slider.setRange(5, 30)
        self.s_slider = QSlider(Qt.Orientation.Horizontal); self.s_slider.setRange(2, 20)
        self.t_slider = QSlider(Qt.Orientation.Horizontal); self.t_slider.setRange(1, 100)
        self.h_label = QLabel(); self.v_label = QLabel(); self.s_label = QLabel(); self.t_label = QLabel()

        self.general_title = QLabel(objectName="sectionTitle")
        self.row_language = QLabel(); self.row_theme = QLabel(); self.row_primary = QLabel(); self.row_h = QLabel(); self.row_v = QLabel(); self.row_s = QLabel(); self.row_t = QLabel()
        gf.addRow(self.general_title)
        gf.addRow(self.row_language, lang_layout)
        gf.addRow(self.row_theme, theme_layout)
        gf.addRow(self.row_primary, hand_layout)
        gf.addRow(self.row_h, self._wrap_slider(self.h_slider, self.h_label))
        gf.addRow(self.row_v, self._wrap_slider(self.v_slider, self.v_label))
        gf.addRow(self.row_s, self._wrap_slider(self.s_slider, self.s_label))
        gf.addRow(self.row_t, self._wrap_slider(self.t_slider, self.t_label))
        left.addWidget(self.general_card)

        self.binding_card = QFrame(objectName="panelCard")
        bf = QFormLayout(self.binding_card)
        self.thumb_index_combo = QComboBox(); self.thumb_middle_combo = QComboBox()
        self.binding_title = QLabel(objectName="sectionTitle")
        self.row_thumb_index = QLabel(); self.row_thumb_middle = QLabel()
        bf.addRow(self.binding_title)
        bf.addRow(self.row_thumb_index, self.thumb_index_combo)
        bf.addRow(self.row_thumb_middle, self.thumb_middle_combo)
        left.addWidget(self.binding_card); left.addStretch()

        self.advanced_card = QFrame(objectName="panelCard")
        af = QFormLayout(self.advanced_card)
        self.move_interval_slider = QSlider(Qt.Orientation.Horizontal); self.move_interval_slider.setRange(8, 40)
        self.scroll_interval_slider = QSlider(Qt.Orientation.Horizontal); self.scroll_interval_slider.setRange(16, 80)
        self.scroll_deadzone_slider = QSlider(Qt.Orientation.Horizontal); self.scroll_deadzone_slider.setRange(6, 30)
        self.scroll_scale_slider = QSlider(Qt.Orientation.Horizontal); self.scroll_scale_slider.setRange(600, 3200)
        self.move_label = QLabel(); self.scr_i_label = QLabel(); self.scr_d_label = QLabel(); self.scr_s_label = QLabel()
        self.perf_title = QLabel(objectName="sectionTitle")
        self.row_move = QLabel(); self.row_scr_i = QLabel(); self.row_scr_d = QLabel(); self.row_scr_s = QLabel()
        af.addRow(self.perf_title)
        af.addRow(self.row_move, self._wrap_slider(self.move_interval_slider, self.move_label))
        af.addRow(self.row_scr_i, self._wrap_slider(self.scroll_interval_slider, self.scr_i_label))
        af.addRow(self.row_scr_d, self._wrap_slider(self.scroll_deadzone_slider, self.scr_d_label))
        af.addRow(self.row_scr_s, self._wrap_slider(self.scroll_scale_slider, self.scr_s_label))
        right.addWidget(self.advanced_card)

        self.note_card = QFrame(objectName="panelCard")
        nl = QVBoxLayout(self.note_card)
        self.notes_title = QLabel(objectName="sectionTitle")
        nl.addWidget(self.notes_title)
        self.note_labels = []
        for _ in range(3):
            lb = QLabel(objectName="hintText"); lb.setWordWrap(True)
            self.note_labels.append(lb); nl.addWidget(lb)
        right.addWidget(self.note_card); right.addStretch()

        self.settings_cards.addWidget(self.left_col, 3)
        self.settings_cards.addWidget(self.right_col, 2)
        self.settings_page.setLayout(self.settings_cards)
        self.settings_scroll.setWidget(self.settings_page)
        self.pages.addWidget(self.settings_scroll)

    def _wrap_slider(self, slider, value_label):
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0)
        value_label.setMinimumWidth(52); value_label.setAlignment(Qt.AlignmentFlag.AlignCenter); value_label.setObjectName("valuePill")
        l.addWidget(slider, 1); l.addWidget(value_label)
        return w

    def _connect_signals(self):
        self.home_btn.clicked.connect(self._handle_nav_click)
        self.settings_btn.clicked.connect(self._handle_nav_click)
        self.toggle_btn.clicked.connect(lambda: self._set_sidebar_expanded(not self.sidebar_expanded))
        self.start_stop_btn.clicked.connect(self._handle_power_click)
        self.lang_group.idClicked.connect(self._on_language_changed_rb)
        self.theme_group.idClicked.connect(self._on_theme_changed_rb)
        self.hand_group.idClicked.connect(lambda: self.apply_preferences())
        for w in [self.h_slider, self.v_slider, self.s_slider, self.t_slider,
                  self.move_interval_slider, self.scroll_interval_slider, self.scroll_deadzone_slider, self.scroll_scale_slider,
                  self.thumb_index_combo, self.thumb_middle_combo]:
            if isinstance(w, QCheckBox):
                w.checkStateChanged.connect(self.apply_preferences)
            else:
                w.currentIndexChanged.connect(self.apply_preferences) if isinstance(w, QComboBox) else w.valueChanged.connect(self.apply_preferences)

    def _on_language_changed_rb(self, id):
        new_lang = "zh" if id == 0 else "en"
        if new_lang != self.language:
            self.language = new_lang
            self.settings["language"] = new_lang
            self.apply_language()
            self.apply_preferences()

    def _on_theme_changed_rb(self, id):
        new_theme = "light" if id == 0 else "dark"
        if new_theme != self.theme_name:
            self.theme_name = new_theme
            self.settings["theme"] = self.theme_name
            save_settings(self.settings)
            self.apply_style()
            self.apply_language()
            self.update()

    def _refresh_combo_labels(self):
        self.thumb_index_combo.blockSignals(True); self.thumb_middle_combo.blockSignals(True)
        t1 = self.thumb_index_combo.currentData() or "left_drag"
        t2 = self.thumb_middle_combo.currentData() or "right_click"
        self.thumb_index_combo.clear(); self.thumb_middle_combo.clear()
        for label, data in [(self._t("g_ld"), "left_drag"), (self._t("g_rc"), "right_click"), (self._t("g_none"), "none")]:
            self.thumb_index_combo.addItem(label, data); self.thumb_middle_combo.addItem(label, data)
        self._set_combo(self.thumb_index_combo, t1); self._set_combo(self.thumb_middle_combo, t2)
        self.thumb_index_combo.blockSignals(False); self.thumb_middle_combo.blockSignals(False)

    def _set_combo(self, combo, data):
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i); break

    def _update_sidebar_text(self):
        if self.sidebar_expanded:
            self.home_btn.setText(f"  🏠  {self._t('nav_home')}")
            self.settings_btn.setText(f"  ⚙  {self._t('nav_settings')}")
            # 页脚文字在展开模式下水平居中显示
            self.footer_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.footer_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.sidebar_version.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.brand_container.show()
        else:
            self.home_btn.setText("🏠")
            self.settings_btn.setText("⚙")
            self.brand_container.hide()

    def apply_language(self):
        self.setWindowTitle(self._t("title"))
        self.toggle_btn.setToolTip(self._t("toggle_sidebar"))
        self.footer_subtitle.setText(self._t("brand_sub"))
        self.hero.setText(self._t("hero")); self.hero_sub.setText(self._t("hero_sub"))
        self.start_stop_btn.setText(self._t("ctrl_stop") if self.start_stop_btn.isChecked() else self._t("ctrl_start"))
        self.status_title_main.setText(self._t("status_main")); self.status_title_helper.setText(self._t("status_helper"))
        self.hint_title.setText(self._t("guide"))
        # self.theme_btn removal
        self.sidebar_version.setText(f"© 2026 nighty • {self._t('version')}")
        for lb, key in zip(self.hint_labels, ["hint_1", "hint_2", "hint_3", "hint_4"]): lb.setText(self._t(key))
        self.general_title.setText(self._t("general")); self.row_language.setText(self._t("language")); self.row_theme.setText(self._t("theme")); self.row_primary.setText(self._t("primary"))
        self.row_h.setText(self._t("h_sens")); self.row_v.setText(self._t("v_sens"))
        self.row_s.setText(self._t("smooth")); self.row_t.setText(self._t("lock"))
        
        # 更新单选框文本
        self.lang_zh_rb.setText(self._t("lang_zh")); self.lang_en_rb.setText(self._t("lang_en"))
        self.theme_light_rb.setText(self._t("theme_light")); self.theme_dark_rb.setText(self._t("theme_dark"))
        self.hand_right_rb.setText(self._t("right")); self.hand_left_rb.setText(self._t("left"))

        self.binding_title.setText(self._t("binding")); self.row_thumb_index.setText(self._t("thumb_index")); self.row_thumb_middle.setText(self._t("thumb_middle"))
        self.perf_title.setText(self._t("perf")); self.row_move.setText(self._t("move_i")); self.row_scr_i.setText(self._t("scroll_i"))
        self.row_scr_d.setText(self._t("scroll_d")); self.row_scr_s.setText(self._t("scroll_s"))
        self.notes_title.setText(self._t("notes"))
        for lb, key in zip(self.note_labels, ["n1", "n2", "n3"]): lb.setText(self._t(key))
        self._refresh_combo_labels()
        self._update_sidebar_text()

    def on_gesture_detected(self, side, name):
        key = STATUS_MAP.get(name, "st_wait")
        label = self.gesture_label_l if side == "left" else self.gesture_label_r
        label.setText(self._t(key))
        self.gesture_timers[side].start(900)

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        
        # 优化：直接使用内容区大小（扣除 padding）进行缩放，确保居中且不超出
        target_w = self.preview_card.width() - 48 # 减去 QFrame 默认内边距
        target_h = self.preview_card.height() - 48
        
        px = QPixmap.fromImage(qt_img).scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(px)

    def load_settings_to_ui(self):
        self._refresh_combo_labels()
        # 更新单选框状态
        self.lang_group.button(0 if self.language == "zh" else 1).setChecked(True)
        self.theme_group.button(0 if self.theme_name == "light" else 1).setChecked(True)
        self.hand_group.button(0 if self.settings["target_hand"] == "Right" else 1).setChecked(True)

        self.h_slider.setValue(int(self.settings["sens_x"] * 10)); self.v_slider.setValue(int(self.settings["sens_y"] * 10))
        self.s_slider.setValue(int(self.settings["smoothing"])); self.t_slider.setValue(int(self.settings["still_threshold"] * 10000))
        self.move_interval_slider.setValue(int(self.settings["move_interval_ms"]))
        self.scroll_interval_slider.setValue(int(self.settings["scroll_interval_ms"]))
        self.scroll_deadzone_slider.setValue(int(self.settings["scroll_deadzone"] * 1000))
        self.scroll_scale_slider.setValue(int(self.settings["scroll_scale"]))
        self._set_combo(self.thumb_index_combo, self.settings["gesture_bindings"]["thumb_index"])
        self._set_combo(self.thumb_middle_combo, self.settings["gesture_bindings"]["thumb_middle"])
        self.gesture_label_l.setText(self._t("waiting")); self.gesture_label_r.setText(self._t("waiting"))
        self.refresh_setting_labels()

    def _percent(self, slider):
        return int(round((slider.value() - slider.minimum()) * 100.0 / max(1, slider.maximum() - slider.minimum())))

    def refresh_setting_labels(self):
        self.h_label.setText(f"{self._percent(self.h_slider)}%"); self.v_label.setText(f"{self._percent(self.v_slider)}%")
        self.s_label.setText(f"{self._percent(self.s_slider)}%"); self.t_label.setText(f"{self._percent(self.t_slider)}%")
        self.move_label.setText(f"{self._percent(self.move_interval_slider)}%")
        self.scr_i_label.setText(f"{self._percent(self.scroll_interval_slider)}%")
        self.scr_d_label.setText(f"{self._percent(self.scroll_deadzone_slider)}%")
        self.scr_s_label.setText(f"{self._percent(self.scroll_scale_slider)}%")

    def apply_preferences(self):
        self.refresh_setting_labels()
        if self.is_loading_ui:
            return
        self.settings["language"] = self.language
        self.settings["target_hand"] = "Right" if self.hand_right_rb.isChecked() else "Left"
        self.settings["sens_x"] = self.h_slider.value() / 10.0
        self.settings["sens_y"] = self.v_slider.value() / 10.0
        self.settings["smoothing"] = float(self.s_slider.value())
        self.settings["still_threshold"] = self.t_slider.value() / 10000.0
        self.settings["move_interval_ms"] = self.move_interval_slider.value()
        self.settings["scroll_interval_ms"] = self.scroll_interval_slider.value()
        self.settings["scroll_deadzone"] = self.scroll_deadzone_slider.value() / 1000.0
        self.settings["scroll_scale"] = self.scroll_scale_slider.value()
        self.settings["gesture_bindings"]["thumb_index"] = self.thumb_index_combo.currentData()
        self.settings["gesture_bindings"]["thumb_middle"] = self.thumb_middle_combo.currentData()
        save_settings(self.settings)
        self.tracker.set_target_hand(self.settings["target_hand"])
        self.tracker.set_sens(self.settings["sens_x"], self.settings["sens_y"])
        self.tracker.set_smoothing(self.settings["smoothing"])
        self.tracker.set_still_threshold(self.settings["still_threshold"])
        self.tracker.set_move_interval_ms(self.settings["move_interval_ms"])
        self.tracker.set_scroll_profile(
            self.settings["scroll_interval_ms"], self.settings["scroll_deadzone"], self.settings["scroll_scale"],
            self.settings["scroll_step_limit"], self.settings["left_scroll_enter_frames"], self.settings["left_scroll_exit_frames"],
        )
        self.tracker.set_gesture_binding("thumb_index", self.settings["gesture_bindings"]["thumb_index"])
        self.tracker.set_gesture_binding("thumb_middle", self.settings["gesture_bindings"]["thumb_middle"])

    def set_page(self, index):
        self.pages.setCurrentIndex(index)
        self.home_btn.setChecked(index == 0); self.settings_btn.setChecked(index == 1)

    def _handle_nav_click(self):
        self.set_page(0 if self.sender() == self.home_btn else 1)

    def _handle_power_click(self):
        is_active = self.start_stop_btn.isChecked()
        self.start_stop_btn.setText(self._t("ctrl_stop") if is_active else self._t("ctrl_start"))
        self.tracker.running = is_active
        if is_active:
            if not self.tracker.isRunning():
                self.tracker.start()
        else:
            # 停止时清空预览图或显示占位
            self.image_label.setText(self._t("waiting"))
            self.gesture_label_l.setText(self._t("waiting"))
            self.gesture_label_r.setText(self._t("waiting"))

    def _set_sidebar_expanded(self, expanded, animate=False):
        if self.sidebar_expanded == expanded:
            return
        self.sidebar_expanded = expanded
        self._update_sidebar_text()
        target = SIDEBAR_WIDE if expanded else SIDEBAR_NARROW
        self.sidebar.setMinimumWidth(target)
        self.sidebar.setMaximumWidth(target)

    def _set_compact_mode(self, compact, animate=False):
        if self.compact_mode == compact:
            return
        self.compact_mode = compact
        self.home_cards.setDirection(QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight)
        self.settings_cards.setDirection(QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight)
        # 当主页面竖直布局时，把两个手势显示控件并排，并平分宽度
        if compact:
            self.status_layout.setDirection(QBoxLayout.Direction.LeftToRight)
            self.status_panel.setMaximumHeight(200) # 限制垂直占用
        else:
            self.status_layout.setDirection(QBoxLayout.Direction.TopToBottom)
            self.status_panel.setMaximumHeight(16777215)

    def _apply_responsive(self, animate=True):
        w = self.width()
        # 同步侧边栏收缩与竖向布局切换的时机 (RESPONSIVE_THRESHOLD = 960)
        should_compact = w < RESPONSIVE_THRESHOLD
        
        if self.sidebar_expanded != (not should_compact):
            self._set_sidebar_expanded(not should_compact, animate=animate)
        
        if self.compact_mode != should_compact:
            self._set_compact_mode(should_compact, animate=animate)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "resize_timer"):
            self.resize_timer.start(80)

    def apply_style(self):
        theme = THEMES[self.theme_name]
        self.setStyleSheet(
            f"""
            * {{ 
                font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei"; 
                outline: none;
            }}
            #mainContainer {{
                background: {theme['content_bg']};
            }}
            QFrame#sidebar {{ 
                background: {theme['sidebar_bg']}; 
                border-right: 1px solid {theme['border']}; 
            }}
            QFrame#contentArea {{ 
                background: {theme['content_bg']}; 
            }}
            QLabel {{ color: {theme['text_primary']}; }}
            QLabel#brandTitle {{ font-size: 24px; font-weight: 800; color: {theme['accent']}; margin-top: 10px; }}
            QLabel#brandSubtitle {{ font-size: 12px; color: {theme['text_secondary']}; letter-spacing: 0.5px; }}
            
            QPushButton#toggleButton {{ 
                background: transparent; border: none; border-radius: 12px; font-size: 20px; padding: 10px; color: {theme['text_secondary']};
            }}
            QPushButton#toggleButton:hover {{ 
                background: {theme['hover']}; color: {theme['text_primary']};
            }}
            
            QPushButton#navButton {{ 
                border: none; border-radius: 16px; padding: 14px; text-align: center; font-size: 18px; font-weight: 600; 
                color: {theme['text_secondary']}; margin: 4px 6px;
            }}
            QPushButton#navButton:hover {{ background: {theme['hover']}; color: {theme['text_primary']}; }}
            QPushButton#navButton:checked {{ background: {theme['accent']}; color: white; }}
            
            QPushButton#powerButton {{
                background: {theme['accent']}; color: white; border-radius: 12px; font-size: 16px; font-weight: 700; border: none;
            }}
            QPushButton#powerButton:checked {{
                background: #34c759; /* Green for active */
            }}
            QPushButton#powerButton:!checked {{
                background: {theme['accent'] if self.theme_name == 'light' else '#38383a'};
                color: {'white' if self.theme_name == 'light' else '#ebebf5'};
            }}
            QPushButton#powerButton:hover {{
                opacity: 0.85;
            }}
            
            QFrame#panelCard {{ 
                background: {theme['card_bg']}; border: 1px solid {theme['border']}; border-radius: 16px; padding: 24px; 
            }}
            
            /* 修正深色模式下背景层级渲染问题 */
            QStackedWidget#stackedPages, 
            QScrollArea#settingsScroll, 
            QScrollArea#settingsScroll > QWidget, 
            QWidget#settingsPage,
            QFrame#contentArea {{
                background: transparent;
                border: none;
            }}
            
            QLabel#heroTitle {{ font-size: 36px; font-weight: 800; color: {theme['text_primary']}; margin-bottom: 2px; }}
            QLabel#heroSub {{ font-size: 16px; color: {theme['text_secondary']}; }}
            QLabel#sectionTitle {{ font-size: 19px; font-weight: 700; color: {theme['text_primary']}; margin-bottom: 12px; }}
            QLabel#hintText {{ color: {theme['text_secondary']}; font-size: 14px; }}
            
            QLabel#previewSurface {{ 
                background: {theme['preview_bg']}; border-radius: 16px; 
            }}
            QLabel#gestureDisplay {{ min-height: 52px; font-size: 22px; font-weight: 700; color: {theme['accent']}; }}
            
            QLabel#valuePill {{ 
                background: {theme['hover']}; border-radius: 10px; padding: 5px 10px; font-size: 12px; font-weight: 600; color: {theme['text_primary']};
            }}
            
            QLabel#sidebarInfoTitle {{ font-size: 13px; font-weight: 700; color: {theme['text_primary']}; }}
            QLabel#sidebarInfoVersion {{ font-size: 11px; color: {theme['text_secondary']}; }}
            
            QComboBox {{ 
                background: {theme['input_bg']}; border: 1px solid {theme['border']}; padding: 8px 14px; font-size: 14px; min-height: 36px; border-radius: 10px;
            }}
            QComboBox:hover {{ border-color: {theme['accent']}; }}
            QComboBox::drop-down {{ border: none; padding-right: 10px; }}
            
            QRadioButton, QCheckBox {{ 
                spacing: 8px; 
                color: {theme['text_primary']}; 
                font-size: 14px;
            }}
            QRadioButton::indicator, QCheckBox::indicator {{ 
                width: 20px; 
                height: 20px; 
                border-radius: 6px; /* 方形圆角 */
                border: 2px solid {theme['border']}; 
                background: {theme['input_bg']}; 
            }}
            QRadioButton::indicator:checked, QCheckBox::indicator:checked {{ 
                background: {theme['accent']}; 
                border-color: {theme['accent']};
            }}
            QRadioButton::indicator:hover, QCheckBox::indicator:hover {{ 
                border-color: {theme['accent']}; 
            }}
            
            QSlider::groove:horizontal {{
                border: none; height: 6px; background: {theme['hover']}; margin: 2px 0; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {theme['accent']}; border: 2px solid {theme['card_bg']}; width: 18px; height: 18px; margin: -7px 0; border-radius: 10px;
            }}
            
            QScrollBar:vertical {{ border: none; background: transparent; width: 6px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {theme['border']}; min-height: 40px; border-radius: 3px; }}
            QScrollBar::handle:vertical:hover {{ background: {theme['text_secondary']}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            """
        )

    def closeEvent(self, event):
        self.tracker.running = False
        self.tracker.wait()
        event.accept()

SIDEBAR_WIDE = 260
SIDEBAR_NARROW = 70
AUTO_COLLAPSE_W = 800
AUTO_EXPAND_W = 1000
COMPACT_W = 750
REGULAR_W = 950
MIN_W = 400
MIN_H = 500
