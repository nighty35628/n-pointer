import pathlib

import cv2

from PyQt6.QtCore import QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, QSize, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
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
)

from src.core.tracker import HandTrackerThread
from src.utils.config import load_settings, save_settings


class NoWheelComboBox(QComboBox):
    """QComboBox variant that ignores mouse wheel events to avoid accidental changes."""
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSlider(QSlider):
    """QSlider variant that ignores mouse wheel events to avoid accidental changes."""
    def wheelEvent(self, event):
        event.ignore()


SIDEBAR_WIDE = 210
SIDEBAR_NARROW = 70
RESPONSIVE_THRESHOLD = 900
MIN_W = 360
MIN_H = 440
PREVIEW_RATIO = 16 / 9
PREVIEW_BASE_W = 720
PREVIEW_BASE_H = 405

I18N = {
    "zh": {
        "title": "N-Pointer",
        "version": "v1.0.2",
        "brand_sub": "手势鼠标控制器",
        "nav_home": "实时控制",
        "nav_settings": "偏好设置",
        "toggle_sidebar": "展开/收起侧栏",
        "hero": "欢迎使用 N-Pointer",
        "hero_sub": "界面布局参考 C-sorting，默认进入静止预览，启动后再进入实时控制。",
        "status_main": "主控手状态",
        "status_helper": "辅助手状态",
        "waiting": "等待识别",
        "guide": "当前说明",
        "hint_1": "主控手剑指控制鼠标移动。",
        "hint_2": "主控手张开保持可将光标复位到中心。",
        "hint_3": "拇指+食指、拇指+中指支持动作映射。",
        "hint_4": "辅助手剑指用于滚动控制。",
        "general": "基础设置",
        "language": "语言",
        "theme": "主题",
        "primary": "主控手",
        "h_sens": "水平灵敏度",
        "v_sens": "垂直灵敏度",
        "smooth": "平滑度",
        "lock": "锁定阈值",
        "binding": "动作映射",
        "thumb_index": "拇指 + 食指",
        "thumb_middle": "拇指 + 中指",
        "perf": "性能参数",
        "perf_lock": "锁定",
        "move_i": "移动间隔 (ms)",
        "scroll_i": "滚动间隔 (ms)",
        "scroll_d": "滚动死区",
        "scroll_s": "滚动倍率",
        "notes": "说明",
        "n1": "窄窗口下会自动切换为紧凑布局。",
        "n2": "预览背景始终保持 16:9 比例，不再随窗口无限拉伸。",
        "n3": "状态卡片改为更扁平的悬浮感样式。",
        "theme_light": "浅色模式",
        "theme_dark": "深色模式",
        "right": "右手",
        "left": "左手",
        "lang_zh": "中文",
        "lang_en": "English",
        "g_ld": "左键",
        "g_rc": "右键",
        "g_none": "无动作",
        "st_wait": "等待识别",
        "st_ld": "左键",
        "st_rc": "右键",
        "st_model": "模型不可用",
        "st_cam": "摄像头不可用",
        "st_show": "显示桌面",
        "st_reset": "中心复位",
        "st_lock": "移动锁定",
        "st_move": "正在移动",
        "st_scroll": "正在滚动",
        "st_scroll_arm": "滚动预备",
        "ctrl_start": "启动控制",
        "ctrl_stop": "停止控制",
        "control_label": "控制开关",
        "control_desc": "默认不启动，先确认姿态与环境，再手动开始实时识别。",
    },
    "en": {
        "title": "N-Pointer",
        "version": "v1.0.2",
        "brand_sub": "Hand gesture mouse controller",
        "nav_home": "Live Control",
        "nav_settings": "Preferences",
        "toggle_sidebar": "Toggle sidebar",
        "hero": "Welcome to N-Pointer",
        "hero_sub": "The layout now follows C-sorting and stays idle until you start live control.",
        "status_main": "Main hand status",
        "status_helper": "Helper hand status",
        "waiting": "Waiting",
        "guide": "Guide",
        "hint_1": "The main-hand sword pose moves the cursor.",
        "hint_2": "Holding an open hand resets the cursor to center.",
        "hint_3": "Thumb+index and thumb+middle actions are configurable.",
        "hint_4": "The helper-hand sword pose controls scrolling.",
        "general": "General",
        "language": "Language",
        "theme": "Theme",
        "primary": "Primary hand",
        "h_sens": "Horizontal sensitivity",
        "v_sens": "Vertical sensitivity",
        "smooth": "Smoothing",
        "lock": "Lock threshold",
        "binding": "Gesture binding",
        "thumb_index": "Thumb + index",
        "thumb_middle": "Thumb + middle",
        "perf": "Performance",
        "perf_lock": "Lock",
        "move_i": "Move interval (ms)",
        "scroll_i": "Scroll interval (ms)",
        "scroll_d": "Scroll deadzone",
        "scroll_s": "Scroll scale",
        "notes": "Notes",
        "n1": "Narrow widths switch the page into a compact layout.",
        "n2": "The preview keeps a fixed 16:9 ratio and no longer stretches with the window.",
        "n3": "Status cards now use a flatter, lighter floating style.",
        "theme_light": "Light Mode",
        "theme_dark": "Dark Mode",
        "right": "Right hand",
        "left": "Left hand",
        "lang_zh": "Chinese",
        "lang_en": "English",
        "g_ld": "Left drag",
        "g_rc": "Right click",
        "g_none": "None",
        "st_wait": "Waiting",
        "st_ld": "Left drag",
        "st_rc": "Right click",
        "st_model": "Model unavailable",
        "st_cam": "Camera unavailable",
        "st_show": "Show desktop",
        "st_reset": "Center reset",
        "st_lock": "Move locked",
        "st_move": "Moving",
        "st_scroll": "Scrolling",
        "st_scroll_arm": "Scroll armed",
        "ctrl_start": "Start Control",
        "ctrl_stop": "Stop Control",
        "control_label": "Power",
        "control_desc": "Live control is off by default so you can check posture and framing first.",
    },
}

THEMES = {
    "light": {
        "window_bg": "#f6f4ef",
        "sidebar_bg": "#ece8df",
        "content_bg": "#fbfaf7",
        "card_bg": "#ffffff",
        "card_soft": "#f4f1ea",
        "text_primary": "#161514",
        "text_secondary": "#706a61",
        "accent": "#fa2d48",
        "accent_soft": "#ffe5e9",
        "border": "#ddd6cb",
        "hover": "#e4ddd2",
        "input_bg": "#ffffff",
    },
    "dark": {
        "window_bg": "#111214",
        "sidebar_bg": "#191b1f",
        "content_bg": "#111214",
        "card_bg": "#1a1d22",
        "card_soft": "#23272e",
        "text_primary": "#f4f5f7",
        "text_secondary": "#9aa0a8",
        "accent": "#ff5a72",
        "accent_soft": "#38252b",
        "border": "#2f343c",
        "hover": "#262b32",
        "input_bg": "#20242b",
    },
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
        self.theme_name = self.settings.get("theme", "light") if self.settings.get("theme") in THEMES else "light"
        self.sidebar_expanded = True
        self.compact_mode = False
        self.is_loading_ui = True
        self.tracker = None
        self.default_preview = self._load_default_preview()
        self.current_preview = None

        self.setWindowTitle(self._t("title"))
        self.setMinimumSize(MIN_W, MIN_H)
        self.resize(800, 600)
        self._init_ui()
        self._init_sidebar_animation()
        self._connect_signals()
        self.load_settings_to_ui()
        self.set_page(0)
        self.apply_style()
        self.apply_language()
        self.is_loading_ui = False
        self.apply_preferences()
        self._apply_responsive(animate=False)
        self._show_default_preview()

    def _t(self, key):
        return I18N[self.language][key]

    def _load_default_preview(self):
        bg_path = pathlib.Path(__file__).resolve().parents[2] / "assets" / "bg.png"
        if not bg_path.exists():
            return QPixmap()
        return QPixmap(str(bg_path))

    def _init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)

        self.main_container = QFrame(objectName="mainContainer")
        self.main_layout = QHBoxLayout(self.main_container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        root_layout.addWidget(self.main_container)

        self.sidebar = QFrame(objectName="sidebar")
        self.sidebar.setProperty("collapsed", "false")
        self.sidebar.setMinimumWidth(SIDEBAR_WIDE)
        self.sidebar.setMaximumWidth(SIDEBAR_WIDE)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 6, 0, 8)
        sidebar_layout.setSpacing(2)

        self.toggle_btn = QPushButton("☰", objectName="toggleButton")
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setFixedSize(48, 48)
        sidebar_layout.addWidget(self.toggle_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(6, 6, 6, 0)
        nav_layout.setSpacing(4)
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.home_btn = QPushButton(objectName="navButton")
        self.home_btn.setCheckable(True)
        self.home_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn = QPushButton(objectName="navButton")
        self.settings_btn.setCheckable(True)
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        for btn in (self.home_btn, self.settings_btn):
            btn.setFixedHeight(48)
            nav_layout.addWidget(btn)
        sidebar_layout.addWidget(nav_container)
        sidebar_layout.addStretch()

        self.brand_container = QWidget()
        brand_layout = QVBoxLayout(self.brand_container)
        brand_layout.setContentsMargins(8, 0, 8, 2)
        brand_layout.setSpacing(2)
        brand_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.footer_title = QLabel("N-POINTER", objectName="brandTitle")
        self.footer_subtitle = QLabel(objectName="brandSubtitle")
        self.sidebar_version = QLabel(objectName="sidebarInfoVersion")
        self.footer_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.footer_subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.sidebar_version.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        brand_layout.addWidget(self.footer_title)
        brand_layout.addWidget(self.footer_subtitle)
        brand_layout.addWidget(self.sidebar_version)
        sidebar_layout.addWidget(self.brand_container)

        self.content = QFrame(objectName="contentArea")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(24, 12, 24, 12)
        content_layout.setSpacing(0)
        self.pages = QStackedWidget(objectName="stackedPages")
        self._init_home_page()
        self._init_settings_page()
        content_layout.addWidget(self.pages)

        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.content, 1)

        self.gesture_timers = {"left": QTimer(self), "right": QTimer(self)}
        self.gesture_timers["left"].setSingleShot(True)
        self.gesture_timers["right"].setSingleShot(True)
        self.gesture_timers["left"].timeout.connect(lambda: self.gesture_label_l.setText(self._t("waiting")))
        self.gesture_timers["right"].timeout.connect(lambda: self.gesture_label_r.setText(self._t("waiting")))

    def _init_sidebar_animation(self):
        self.sidebar_anim = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self.sidebar_anim_max = QPropertyAnimation(self.sidebar, b"maximumWidth")
        for anim in (self.sidebar_anim, self.sidebar_anim_max):
            anim.setDuration(280)
            anim.setEasingCurve(QEasingCurve.Type.OutQuint)
        self.sidebar_anim_group = QParallelAnimationGroup(self)
        self.sidebar_anim_group.addAnimation(self.sidebar_anim)
        self.sidebar_anim_group.addAnimation(self.sidebar_anim_max)

    def _init_home_page(self):
        self.home_page = QWidget()
        layout = QVBoxLayout(self.home_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        hero_box = QVBoxLayout()
        hero_box.setSpacing(2)
        self.hero = QLabel(objectName="heroTitle")
        self.hero_sub = QLabel(objectName="heroSub")
        self.hero_sub.setWordWrap(True)
        hero_box.addWidget(self.hero)
        hero_box.addWidget(self.hero_sub)
        top_bar.addLayout(hero_box, 1)
        layout.addLayout(top_bar)

        self.home_cards = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.home_cards.setSpacing(10)

        self.preview_card = QFrame(objectName="previewCard")
        preview_layout = QVBoxLayout(self.preview_card)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(0)
        self.preview_holder = QWidget(objectName="previewHolder")
        holder_layout = QVBoxLayout(self.preview_holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        holder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel(objectName="previewSurface")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.image_label.setMinimumSize(320, 180)
        holder_layout.addWidget(self.image_label, 0, Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_holder)

        self.side_panel = QWidget()
        self.side_panel_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, self.side_panel)
        self.side_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.side_panel_layout.setSpacing(8)

        self.control_card = QFrame(objectName="controlCard")
        control_layout = QVBoxLayout(self.control_card)
        control_layout.setContentsMargins(8, 0, 8, 8)
        control_layout.setSpacing(6)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.control_label = QLabel(objectName="statusEyebrow")
        self.control_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_desc = QLabel(objectName="controlText")
        self.control_desc.setWordWrap(True)
        self.start_stop_btn = QPushButton(objectName="powerButton")
        self.start_stop_btn.setCheckable(True)
        self.start_stop_btn.setChecked(False)
        self.start_stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_stop_btn.setFixedSize(120, 44)
        control_layout.addStretch()
        control_layout.addWidget(self.control_label)
        control_layout.addWidget(self.start_stop_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        control_layout.addStretch()

        self.status_title_main, self.gesture_label_r, self.main_status_card = self._create_status_card()
        self.status_title_helper, self.gesture_label_l, self.helper_status_card = self._create_status_card()

        self.side_panel_layout.addWidget(self.control_card)
        self.side_panel_layout.addWidget(self.main_status_card)
        self.side_panel_layout.addWidget(self.helper_status_card)
        self.side_panel_layout.setStretch(0, 6)
        self.side_panel_layout.setStretch(1, 5)
        self.side_panel_layout.setStretch(2, 5)

        # 侧栏卡片宽度将在侧栏展开/折叠时动态设置，初始化不固定以允许响应式布局

        self.home_cards.addWidget(self.preview_card, 7)
        self.home_cards.addWidget(self.side_panel, 3)
        layout.addLayout(self.home_cards, 1)
        self.pages.addWidget(self.home_page)

    def _create_status_card(self):
        card = QFrame(objectName="statusCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        label = QLabel("HAND STATUS", objectName="statusEyebrow")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel(objectName="statusTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_wrap = QFrame(objectName="statusValueWrap")
        value_layout = QVBoxLayout(value_wrap)
        value_layout.setContentsMargins(8, 6, 8, 6)
        value_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        value = QLabel(objectName="gestureDisplay")
        value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_layout.addWidget(value)
        layout.addWidget(label)
        layout.addWidget(title)
        layout.addWidget(value_wrap)
        layout.addStretch()
        return title, value, card

    def _init_settings_page(self):
        self.settings_scroll = QScrollArea(objectName="settingsScroll")
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.settings_page = QWidget(objectName="settingsPage")
        self.settings_cards = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.settings_cards.setContentsMargins(0, 0, 0, 0)
        self.settings_cards.setSpacing(10)

        self.left_col = QWidget()
        left = QVBoxLayout(self.left_col)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)

        self.right_col = QWidget()
        right = QVBoxLayout(self.right_col)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)

        self.general_card = QFrame(objectName="panelCard")
        gf = QFormLayout(self.general_card)
        gf.setContentsMargins(12, 12, 12, 12)
        gf.setHorizontalSpacing(16)
        gf.setVerticalSpacing(12)

        self.lang_group = QButtonGroup(self)
        self.lang_zh_rb = QRadioButton()
        self.lang_en_rb = QRadioButton()
        self.lang_group.addButton(self.lang_zh_rb, 0)
        self.lang_group.addButton(self.lang_en_rb, 1)
        lang_layout = QHBoxLayout()
        lang_layout.setSpacing(12)
        lang_layout.addWidget(self.lang_zh_rb)
        lang_layout.addWidget(self.lang_en_rb)
        lang_layout.addStretch()

        self.theme_group = QButtonGroup(self)
        self.theme_light_rb = QRadioButton()
        self.theme_dark_rb = QRadioButton()
        self.theme_group.addButton(self.theme_light_rb, 0)
        self.theme_group.addButton(self.theme_dark_rb, 1)
        theme_layout = QHBoxLayout()
        theme_layout.setSpacing(12)
        theme_layout.addWidget(self.theme_light_rb)
        theme_layout.addWidget(self.theme_dark_rb)
        theme_layout.addStretch()

        self.hand_group = QButtonGroup(self)
        self.hand_right_rb = QRadioButton()
        self.hand_left_rb = QRadioButton()
        self.hand_group.addButton(self.hand_right_rb, 0)
        self.hand_group.addButton(self.hand_left_rb, 1)
        hand_layout = QHBoxLayout()
        hand_layout.setSpacing(12)
        hand_layout.addWidget(self.hand_right_rb)
        hand_layout.addWidget(self.hand_left_rb)
        hand_layout.addStretch()

        self.h_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.h_slider.setRange(5, 30)
        self.v_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.v_slider.setRange(5, 30)
        self.s_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.s_slider.setRange(2, 20)
        self.t_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.t_slider.setRange(1, 100)
        self.h_label = QLabel()
        self.v_label = QLabel()
        self.s_label = QLabel()
        self.t_label = QLabel()

        self.general_title = QLabel(objectName="sectionTitle")
        self.row_language = QLabel()
        self.row_theme = QLabel()
        self.row_primary = QLabel()
        self.row_h = QLabel()
        self.row_v = QLabel()
        self.row_s = QLabel()
        self.row_t = QLabel()
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
        bf.setContentsMargins(12, 12, 12, 12)
        bf.setHorizontalSpacing(16)
        bf.setVerticalSpacing(12)
        self.thumb_index_combo = NoWheelComboBox()
        self.thumb_middle_combo = NoWheelComboBox()
        self.binding_title = QLabel(objectName="sectionTitle")
        self.row_thumb_index = QLabel()
        self.row_thumb_middle = QLabel()
        bf.addRow(self.binding_title)
        bf.addRow(self.row_thumb_index, self.thumb_index_combo)
        bf.addRow(self.row_thumb_middle, self.thumb_middle_combo)
        left.addWidget(self.binding_card)
        left.addStretch()

        self.advanced_card = QFrame(objectName="panelCard")
        af = QFormLayout(self.advanced_card)
        af.setContentsMargins(12, 12, 12, 12)
        af.setHorizontalSpacing(16)
        af.setVerticalSpacing(12)
        self.move_interval_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.move_interval_slider.setRange(8, 40)
        self.scroll_interval_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.scroll_interval_slider.setRange(16, 80)
        self.scroll_deadzone_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.scroll_deadzone_slider.setRange(6, 30)
        self.scroll_scale_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.scroll_scale_slider.setRange(600, 3200)
        self.move_label = QLabel()
        self.scr_i_label = QLabel()
        self.scr_d_label = QLabel()
        self.scr_s_label = QLabel()
        self.perf_title = QLabel(objectName="sectionTitle")
        self.row_move = QLabel()
        self.row_scr_i = QLabel()
        self.row_scr_d = QLabel()
        self.row_scr_s = QLabel()
        self.perf_lock_cb = QCheckBox()
        self.perf_lock_cb.setChecked(True)
        af.addRow(self.perf_title, self.perf_lock_cb)
        af.addRow(self.row_move, self._wrap_slider(self.move_interval_slider, self.move_label))
        af.addRow(self.row_scr_i, self._wrap_slider(self.scroll_interval_slider, self.scr_i_label))
        af.addRow(self.row_scr_d, self._wrap_slider(self.scroll_deadzone_slider, self.scr_d_label))
        af.addRow(self.row_scr_s, self._wrap_slider(self.scroll_scale_slider, self.scr_s_label))
        right.addWidget(self.advanced_card)

        self.hint_card = QFrame(objectName="panelCard")
        hl = QVBoxLayout(self.hint_card)
        hl.setContentsMargins(12, 12, 12, 12)
        hl.setSpacing(8)
        self.hint_title = QLabel(objectName="sectionTitle")
        hl.addWidget(self.hint_title)
        self.hint_labels = []
        for _ in range(7):
            lb = QLabel(objectName="hintText")
            lb.setWordWrap(True)
            self.hint_labels.append(lb)
            hl.addWidget(lb)
        right.addWidget(self.hint_card)
        right.addStretch()

        self.settings_cards.addWidget(self.left_col, 3)
        self.settings_cards.addWidget(self.right_col, 2)
        self.settings_page.setLayout(self.settings_cards)
        self.settings_scroll.setWidget(self.settings_page)
        self.pages.addWidget(self.settings_scroll)

    def _wrap_slider(self, slider, value_label):
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        value_label.setMinimumWidth(56)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setObjectName("valuePill")
        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return wrapper

    def _connect_signals(self):
        self.home_btn.clicked.connect(self._handle_nav_click)
        self.settings_btn.clicked.connect(self._handle_nav_click)
        self.toggle_btn.clicked.connect(lambda: self._set_sidebar_expanded(not self.sidebar_expanded, animate=True))
        self.start_stop_btn.clicked.connect(self._handle_power_click)
        self.lang_group.idClicked.connect(self._on_language_changed_rb)
        self.perf_lock_cb.toggled.connect(self._toggle_perf_lock)
        self.theme_group.idClicked.connect(self._on_theme_changed_rb)
        self.hand_group.idClicked.connect(lambda: self.apply_preferences())
        for widget in [
            self.h_slider,
            self.v_slider,
            self.s_slider,
            self.t_slider,
            self.move_interval_slider,
            self.scroll_interval_slider,
            self.scroll_deadzone_slider,
            self.scroll_scale_slider,
            self.thumb_index_combo,
            self.thumb_middle_combo,
        ]:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.apply_preferences)
            else:
                widget.valueChanged.connect(self.apply_preferences)

    def _on_language_changed_rb(self, idx):
        new_lang = "zh" if idx == 0 else "en"
        if new_lang != self.language:
            self.language = new_lang
            self.apply_language()
            self.apply_preferences()

    def _on_theme_changed_rb(self, idx):
        new_theme = "light" if idx == 0 else "dark"
        if new_theme != self.theme_name:
            self.theme_name = new_theme
            self.settings["theme"] = new_theme
            save_settings(self.settings)
            self.apply_style()
            self.apply_language()

    def _toggle_perf_lock(self, locked):
        widgets = [
            self.move_interval_slider, self.scroll_interval_slider, 
            self.scroll_deadzone_slider, self.scroll_scale_slider,
            self.row_move, self.row_scr_i, self.row_scr_d, self.row_scr_s,
            self.move_label, self.scr_i_label, self.scr_d_label, self.scr_s_label
        ]
        for w in widgets:
            w.setEnabled(not locked)

    def _refresh_combo_labels(self):
        combos = [
            (self.thumb_index_combo, self.thumb_index_combo.currentData() or "left_drag"),
            (self.thumb_middle_combo, self.thumb_middle_combo.currentData() or "right_click"),
        ]
        options = [
            (self._t("g_ld"), "left_drag"),
            (self._t("g_rc"), "right_click"),
            (self._t("g_none"), "none"),
        ]
        for combo, current in combos:
            combo.blockSignals(True)
            combo.clear()
            for label, data in options:
                combo.addItem(label, data)
            self._set_combo(combo, current)
            combo.blockSignals(False)

    def _set_combo(self, combo, data):
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _update_sidebar_text(self):
        if self.sidebar_expanded:
            self.home_btn.setText(f"🏠 {self._t('nav_home')}")
            self.settings_btn.setText(f"⚙ {self._t('nav_settings')}")
            self.home_btn.setFixedWidth(186)
            self.settings_btn.setFixedWidth(186)
            self.brand_container.show()
        else:
            self.home_btn.setText("🏠")
            self.settings_btn.setText("⚙")
            self.home_btn.setFixedWidth(48)
            self.settings_btn.setFixedWidth(48)
            self.brand_container.hide()
        self.sidebar.setProperty("collapsed", "false" if self.sidebar_expanded else "true")
        self.sidebar.style().unpolish(self.sidebar)
        self.sidebar.style().polish(self.sidebar)

    def apply_language(self):
        self.setWindowTitle(self._t("title"))
        self.toggle_btn.setToolTip(self._t("toggle_sidebar"))
        self.footer_subtitle.setText(self._t("brand_sub"))
        self.sidebar_version.setText(f"© 2026 nighty · {self._t('version')}")
        self.hero.setText(self._t("hero"))
        self.hero_sub.setText(self._t("hero_sub"))
        self.control_label.setText(self._t("control_label"))
        self.start_stop_btn.setText(self._t("ctrl_stop") if self.start_stop_btn.isChecked() else self._t("ctrl_start"))
        self.status_title_main.setText(self._t("status_main"))
        self.status_title_helper.setText(self._t("status_helper"))
        self.hint_title.setText(self._t("guide"))
        self.general_title.setText(self._t("general"))
        self.row_language.setText(self._t("language"))
        self.row_theme.setText(self._t("theme"))
        self.row_primary.setText(self._t("primary"))
        self.row_h.setText(self._t("h_sens"))
        self.row_v.setText(self._t("v_sens"))
        self.row_s.setText(self._t("smooth"))
        self.perf_lock_cb.setText(self._t("perf_lock"))
        self.row_t.setText(self._t("lock"))
        self.binding_title.setText(self._t("binding"))
        self.row_thumb_index.setText(self._t("thumb_index"))
        self.row_thumb_middle.setText(self._t("thumb_middle"))
        self.perf_title.setText(self._t("perf"))
        self.row_move.setText(self._t("move_i"))
        self.row_scr_i.setText(self._t("scroll_i"))
        self.row_scr_d.setText(self._t("scroll_d"))
        self.row_scr_s.setText(self._t("scroll_s"))
        self.hint_title.setText(self._t("guide"))
        self.lang_zh_rb.setText(self._t("lang_zh"))
        self.lang_en_rb.setText(self._t("lang_en"))
        self.theme_light_rb.setText(self._t("theme_light"))
        self.theme_dark_rb.setText(self._t("theme_dark"))
        self.hand_right_rb.setText(self._t("right"))
        self.hand_left_rb.setText(self._t("left"))
        all_hints = ["hint_1", "hint_2", "hint_3", "hint_4", "n1", "n2", "n3"]
        for lb, key in zip(self.hint_labels, all_hints):
            lb.setText(self._t(key))
        self._refresh_combo_labels()
        self._update_sidebar_text()

    def _preview_target_size(self):
        # 允许基础尺寸随窗口缩放，不再限制在固定的 PREVIEW_BASE_W
        available_w = max(320, self.preview_holder.width() - 8)
        available_h = max(180, self.preview_holder.height() - 8)
        
        # 尝试以 16:9 比例填充可用区域的最大宽度
        target_w = available_w
        target_h = int(target_w / PREVIEW_RATIO)
        
        # 如果计算出的高度超出了可用高度，则以高度为基准反推宽度
        if target_h > available_h:
            target_h = available_h
            target_w = int(target_h * PREVIEW_RATIO)
            
        return QSize(max(320, target_w), max(180, target_h))

    def _cover_pixmap(self, pixmap, target_size):
        # 使用更为平滑且比例正确的缩放方式
        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        # 居中裁剪（如果比例不完全匹配）
        x = max(0, (scaled.width() - target_size.width()) // 2)
        y = max(0, (scaled.height() - target_size.height()) // 2)
        return scaled.copy(x, y, target_size.width(), target_size.height())

    def _set_preview_pixmap(self, pixmap):
        if pixmap is None or pixmap.isNull():
            self.image_label.clear()
            return
        target = self._preview_target_size()
        self.image_label.setFixedSize(target)
        self.image_label.setPixmap(self._cover_pixmap(pixmap, target))

    def _show_default_preview(self):
        self.current_preview = None
        self._set_preview_pixmap(self.default_preview)
        self.gesture_label_l.setText(self._t("waiting"))
        self.gesture_label_r.setText(self._t("waiting"))

    def on_gesture_detected(self, side, name):
        key = STATUS_MAP.get(name, "st_wait")
        label = self.gesture_label_l if side == "left" else self.gesture_label_r
        label.setText(self._t(key))
        self.gesture_timers[side].start(900)

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.current_preview = QPixmap.fromImage(qt_img)
        self._set_preview_pixmap(self.current_preview)

    def load_settings_to_ui(self):
        self._refresh_combo_labels()
        self.lang_group.button(0 if self.language == "zh" else 1).setChecked(True)
        self.theme_group.button(0 if self.theme_name == "light" else 1).setChecked(True)
        self.hand_group.button(0 if self.settings["target_hand"] == "Right" else 1).setChecked(True)
        self.h_slider.setValue(int(self.settings["sens_x"] * 10))
        self.v_slider.setValue(int(self.settings["sens_y"] * 10))
        self.s_slider.setValue(int(self.settings["smoothing"]))
        self.t_slider.setValue(int(self.settings["still_threshold"] * 10000))
        self.move_interval_slider.setValue(int(self.settings["move_interval_ms"]))
        self.scroll_interval_slider.setValue(int(self.settings["scroll_interval_ms"]))
        self._toggle_perf_lock(self.perf_lock_cb.isChecked())
        self.scroll_deadzone_slider.setValue(int(self.settings["scroll_deadzone"] * 1000))
        self.scroll_scale_slider.setValue(int(self.settings["scroll_scale"]))
        self._set_combo(self.thumb_index_combo, self.settings["gesture_bindings"]["thumb_index"])
        self._set_combo(self.thumb_middle_combo, self.settings["gesture_bindings"]["thumb_middle"])
        self.gesture_label_l.setText(self._t("waiting"))
        self.gesture_label_r.setText(self._t("waiting"))
        self.refresh_setting_labels()

    def _percent(self, slider):
        return int(round((slider.value() - slider.minimum()) * 100.0 / max(1, slider.maximum() - slider.minimum())))

    def refresh_setting_labels(self):
        self.h_label.setText(f"{self._percent(self.h_slider)}%")
        self.v_label.setText(f"{self._percent(self.v_slider)}%")
        self.s_label.setText(f"{self._percent(self.s_slider)}%")
        self.t_label.setText(f"{self._percent(self.t_slider)}%")
        self.move_label.setText(f"{self._percent(self.move_interval_slider)}%")
        self.scr_i_label.setText(f"{self._percent(self.scroll_interval_slider)}%")
        self.scr_d_label.setText(f"{self._percent(self.scroll_deadzone_slider)}%")
        self.scr_s_label.setText(f"{self._percent(self.scroll_scale_slider)}%")

    def apply_preferences(self):
        self.refresh_setting_labels()
        if self.is_loading_ui:
            return
        self.settings["language"] = self.language
        self.settings["theme"] = self.theme_name
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
        if self.tracker is None or not self.tracker.isRunning():
            return
        self.tracker.set_target_hand(self.settings["target_hand"])
        self.tracker.set_sens(self.settings["sens_x"], self.settings["sens_y"])
        self.tracker.set_smoothing(self.settings["smoothing"])
        self.tracker.set_still_threshold(self.settings["still_threshold"])
        self.tracker.set_move_interval_ms(self.settings["move_interval_ms"])
        self.tracker.set_scroll_profile(
            self.settings["scroll_interval_ms"],
            self.settings["scroll_deadzone"],
            self.settings["scroll_scale"],
            self.settings["scroll_step_limit"],
            self.settings["left_scroll_enter_frames"],
            self.settings["left_scroll_exit_frames"],
        )
        self.tracker.set_gesture_binding("thumb_index", self.settings["gesture_bindings"]["thumb_index"])
        self.tracker.set_gesture_binding("thumb_middle", self.settings["gesture_bindings"]["thumb_middle"])

    def set_page(self, index):
        self.pages.setCurrentIndex(index)
        self.home_btn.setChecked(index == 0)
        self.settings_btn.setChecked(index == 1)

    def _handle_nav_click(self):
        self.set_page(0 if self.sender() == self.home_btn else 1)

    def _ensure_tracker(self):
        if self.tracker is not None and self.tracker.isRunning():
            return
        self.tracker = HandTrackerThread(self.settings)
        self.tracker.change_pixmap_signal.connect(self.update_image)
        self.tracker.gesture_detected_signal.connect(self.on_gesture_detected)

    def _handle_power_click(self):
        is_active = self.start_stop_btn.isChecked()
        self.start_stop_btn.setText(self._t("ctrl_stop") if is_active else self._t("ctrl_start"))
        if is_active:
            self._ensure_tracker()
            self.tracker.running = True
            if not self.tracker.isRunning():
                self.tracker.start()
        else:
            if self.tracker is not None:
                self.tracker.running = False
                self.tracker.wait(1200)
            self._show_default_preview()

    def _set_sidebar_expanded(self, expanded, animate=False):
        if self.sidebar_expanded == expanded:
            return
        self.sidebar_expanded = expanded
        self._update_sidebar_text()
        target = SIDEBAR_WIDE if expanded else SIDEBAR_NARROW
        if animate:
            current = self.sidebar.width()
            self.sidebar_anim_group.stop()
            self.sidebar_anim.setStartValue(current)
            self.sidebar_anim.setEndValue(target)
            self.sidebar_anim_max.setStartValue(current)
            self.sidebar_anim_max.setEndValue(target)
            self.sidebar_anim_group.start()
        else:
            self.sidebar.setMinimumWidth(target)
            self.sidebar.setMaximumWidth(target)
        # 当侧栏展开时固定右侧面板与卡片宽度，折叠时清除固定以恢复响应式布局
        # 注意：在窄模式（compact_mode）下，不应强制固定宽度，否则横向排列会溢出
        try:
            if self.sidebar_expanded and not self.compact_mode:
                self.side_panel.setFixedWidth(160)
                self.control_card.setFixedWidth(140)
                self.main_status_card.setFixedWidth(140)
                self.helper_status_card.setFixedWidth(140)
            else:
                # 清除固定宽度，允许布局自动调整
                self.side_panel.setMinimumWidth(0)
                self.side_panel.setMaximumWidth(16777215)
                self.side_panel.setFixedWidth(16777215) # 显式解除固定
                for w in (self.control_card, self.main_status_card, self.helper_status_card):
                    w.setMinimumWidth(0)
                    w.setMaximumWidth(16777215)
                    w.setFixedWidth(16777215) # 显式解除固定
        except Exception:
            # 如果在初始化阶段这些控件尚未创建，则忽略设置
            pass

    def _set_compact_mode(self, compact):
        if self.compact_mode == compact:
            return
        self.compact_mode = compact
        self.home_cards.setDirection(QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight)
        self.settings_cards.setDirection(QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight)
        self.side_panel_layout.setDirection(QBoxLayout.Direction.LeftToRight if compact else QBoxLayout.Direction.TopToBottom)
        
        # 切换模式时重新触发一次宽度逻辑，确保从宽模式切换到窄模式时解除固定宽度
        self._update_sidebar_text() # 这会间接触发一部分 UI 刷新，但我们需要显式处理卡片
        if compact:
            self.side_panel.setMinimumWidth(0)
            self.side_panel.setMaximumWidth(16777215)
            self.side_panel.setFixedWidth(16777215)
            for w in (self.control_card, self.main_status_card, self.helper_status_card):
                w.setMinimumWidth(0)
                w.setMaximumWidth(16777215)
                w.setFixedWidth(16777215)
        elif self.sidebar_expanded:
            # 如果切回宽模式且侧栏是展开的，恢复固定宽度
            self.side_panel.setFixedWidth(160)
            self.control_card.setFixedWidth(140)
            self.main_status_card.setFixedWidth(140)
            self.helper_status_card.setFixedWidth(140)

    def _apply_responsive(self, animate=True):
        compact = self.width() < RESPONSIVE_THRESHOLD
        self._set_compact_mode(compact)
        self._set_preview_pixmap(self.current_preview or self.default_preview)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive(animate=False)

    def apply_style(self):
        theme = THEMES[self.theme_name]
        self.setStyleSheet(
            f"""
            * {{
                font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei";
                outline: none;
                color: {theme['text_primary']};
            }}
            QMainWindow {{
                background: {theme['window_bg']};
            }}
            QFrame#mainContainer {{
                background: transparent;
                border: none;
            }}
            QFrame#sidebar {{
                background: {theme['sidebar_bg']};
                border-right: 1px solid {theme['border']};
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
            }}
            QFrame#contentArea, QStackedWidget#stackedPages, QScrollArea#settingsScroll, QWidget#settingsPage {{
                background: transparent;
                border: none;
            }}
            QLabel#brandTitle {{
                font-size: 22px;
                font-weight: 800;
                color: {theme['accent']};
            }}
            QLabel#brandSubtitle, QLabel#sidebarInfoVersion {{
                font-size: 12px;
                color: {theme['text_secondary']};
            }}
            QPushButton#toggleButton {{
                border: none;
                border-radius: 14px;
                background: transparent;
                font-size: 20px;
                color: {theme['text_secondary']};
            }}
            QPushButton#toggleButton:hover {{
                background: {theme['hover']};
                color: {theme['text_primary']};
            }}
            QPushButton#navButton {{
                border: none;
                border-radius: 14px;
                background: transparent;
                text-align: center;
                padding: 0 8px;
                font-size: 15px;
                font-weight: 600;
                color: {theme['text_secondary']};
            }}
            QFrame#sidebar[collapsed="true"] QPushButton#navButton {{
                text-align: center;
                padding: 0;
                min-width: 48px;
                max-width: 48px;
            }}
            QPushButton#navButton:hover {{
                background: {theme['hover']};
                color: {theme['text_primary']};
            }}
            QPushButton#navButton:checked {{
                background: {theme['accent']};
                color: white;
            }}
            QLabel#heroTitle {{
                font-size: 34px;
                font-weight: 800;
            }}
            QLabel#heroSub, QLabel#controlText {{
                font-size: 14px;
                color: {theme['text_secondary']};
            }}
            QPushButton#powerButton {{
                background: {theme['accent']};
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 15px;
                font-weight: 700;
                padding: 0 18px;
            }}
            QPushButton#powerButton:checked {{
                background: #2fb36f;
            }}
            QFrame#previewCard, QFrame#controlCard {{
                background: transparent;
                border: none;
                border-radius: 0px;
            }}
            QFrame#panelCard, QFrame#statusCard {{
                background: {theme['card_bg']};
                border: 1px solid {theme['border']};
                border-radius: 22px;
            }}
            QWidget#previewHolder, QLabel#previewSurface {{
                background: transparent;
                border: none;
            }}
            QLabel#statusEyebrow {{
                font-size: 11px;
                font-weight: 700;
                color: {theme['text_secondary']};
                letter-spacing: 1px;
            }}
            QLabel#statusTitle {{
                font-size: 18px;
                font-weight: 700;
            }}
            QFrame#statusValueWrap {{
                background: {theme['card_soft']};
                border: 1px solid {theme['border']};
                border-radius: 18px;
            }}
            QLabel#gestureDisplay {{
                min-height: 32px;
                font-size: 17px;
                font-weight: 700;
                color: {theme['accent']};
            }}
            QLabel#sectionTitle {{
                font-size: 18px;
                font-weight: 700;
            }}
            QLabel#hintText {{
                font-size: 14px;
                color: {theme['text_secondary']};
            }}
            QLabel#valuePill {{
                background: {theme['accent_soft']};
                border-radius: 11px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: 700;
                color: {theme['accent']};
            }}
            QComboBox {{
                background: {theme['input_bg']};
                border: 2px solid {theme['border']};
                border-radius: 12px;
                padding: 4px 12px;
                min-height: 28px;
                font-size: 14px;
            }}
            QComboBox:hover {{
                border-color: {theme['accent']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox QAbstractItemView {{
                background: {theme['card_bg']};
                border: 2px solid {theme['border']};
                border-radius: 12px;
                selection-background-color: {theme['accent_soft']};
                selection-color: {theme['accent']};
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                min-height: 32px;
                padding-left: 10px;
            }}
            QRadioButton {{
                spacing: 8px;
                font-size: 14px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid {theme['border']};
                background: {theme['input_bg']};
            }}
            QRadioButton::indicator:checked {{
                border-color: {theme['accent']};
                background: {theme['accent']};
            }}
            QCheckBox {{
                spacing: 8px;
                font-size: 14px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid {theme['border']};
                background: {theme['input_bg']};
            }}
            QCheckBox::indicator:checked {{
                background: {theme['accent']};
                border-color: {theme['accent']};
                image: none;
            }}
            QCheckBox:disabled, QRadioButton:disabled, QSlider:disabled, QLabel:disabled {{
                color: {theme['text_secondary']};
                opacity: 0.5;
            }}
            QSlider::groove:horizontal:disabled {{
                background: {theme['border']};
            }}
            QSlider::handle:horizontal:disabled {{
                background: {theme['text_secondary']};
            }}
            QCheckBox::indicator:unchecked:hover, QRadioButton::indicator:unchecked:hover {{
                border-color: {theme['accent']};
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                border-radius: 3px;
                background: {theme['card_soft']};
            }}
            QSlider::handle:horizontal {{
                width: 18px;
                margin: -7px 0;
                border-radius: 9px;
                border: 2px solid {theme['card_bg']};
                background: {theme['accent']};
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 8px;
                margin: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme['border']};
                border-radius: 4px;
                min-height: 40px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            """
        )

    def closeEvent(self, event):
        if self.tracker is not None and self.tracker.isRunning():
            self.tracker.running = False
            self.tracker.wait(1200)
        event.accept()
