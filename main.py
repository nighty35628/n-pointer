import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import ControlWindow


def main():
    app = QApplication(sys.argv)
    window = ControlWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
