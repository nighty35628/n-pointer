# N-Pointer (All-in-One 手势鼠标)

N-Pointer 是一个基于计算机视觉的 **全能手势鼠标控制器**。它利用摄像头捕捉用户的手势，并将其转化为鼠标移动、点击、滚动等系统级操作。本项目结合了 MediaPipe 的高效识别能力与 PyQt6 的现代界面设计，为您提供一种流畅、直观的非接触式交互体验。

博文地址: [https://blog.nightytech.com/n-pointer/](https://blog.nightytech.com/n-pointer/)

## 🚀 核心特性

-   **单手/联动操作**：通过单手即可完美控制移动、点击、滚动及复位。支持辅助手进行页面滚动。
-   **现代化 GUI (PyQt6)**：内置实时监控窗口，支持响应式布局、侧边栏导航、中英文多语言切换及深色模式界面。
-   **类鼠标丝滑体验**：内置 1.8x 高灵敏度映射及 3.0 平滑过滤算法，操作精准且无抖动。
-   **智能交互逻辑**：具备移动手势优先机制，防止移动过程中光标误触发点击。
-   **高度可配置**：可在 UI 中实时调节感应灵敏度、平滑度，并保存至本地配置文件。

## 🖐️ 手势操作指南

| 动作 | 手势描述 | 功能说明 |
| :--- | :--- | :--- |
| **光标移动** | **剑指 (食指 + 中指并拢)** | 精度映射至全屏，支持边缘锁定与复位 |
| **左键点击/长按** | **食指 + 大拇指捏合** | 模拟鼠标左键长按与松开 |
| **右键点击** | **中指 + 大拇指捏合** | 触发一次鼠标右键点击 |
| **页面滚动** | **辅助手垂直移动** | 使用非主控手进行页面上下滚动操作 |
| **快速复位** | **张开手掌 (🖐️)** | 将光标快速重置到屏幕中心位置 |

## 🛠️ 技术栈

-   **[MediaPipe](https://github.com/google/mediapipe)**: 用于实时手部关键点检测与三维坐标跟踪。
-   **[OpenCV-Python](https://opencv.org/)**: 负责底层视频流采集与图像预处理。
-   **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)**: 构建高性能、跨平台的桌面交互界面。
-   **[PyAutoGUI](https://github.com/asweigart/pyautogui)**: 模拟跨系统的键盘与鼠标底层操作。
-   **[NumPy](https://numpy.org/)**: 数据矩阵运算与平滑算法支持。

## 📂 项目结构

```text
n-pointer/
├── main.py                # 程序入口，初始化 GUI
├── hand_controller.py     # 手势识别与逻辑控制核心
├── hand_landmarker.task   # MediaPipe 模型文件
├── src/
│   ├── core/
│   │   └── tracker.py     # 异步摄像头采集、手势算法与驱动逻辑
│   ├── ui/
│   │   └── main_window.py # PyQt6 界面逻辑与多语言支持
│   └── utils/
│       └── config.py      # 配置管理与参数计算
└── assets/                # 图标及资源文件
```

## ⚙️ 安装与运行

### 环境准备
-   操作系统：Windows / macOS / Linux (已在 Windows 上测试优化)
-   硬件需求：高清摄像头 (720p 推荐)
-   环境要求：Python 3.8+

### 1. 安装依赖
在项目根目录下运行以下命令安装所需库：
```powershell
pip install -r requirements.txt
```

### 2. 启动应用
执行入口文件启动 GUI：
```powershell
python main.py
```

### 3. 开始使用
1.  启动后在界面左侧点击 **"开始控制"**。
2.  确保光照充足，将手势置于摄像头视野内。
3.  通过侧边栏的 **"设置"** 选项卡可以调整光标灵敏度或切换语言。

---

## 📄 许可证
本项目采用 AGPLv3 许可证。欢迎提交 Issue 或 Pull Request 参与改进！
