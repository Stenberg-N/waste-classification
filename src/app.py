import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QPushButton, QGridLayout, QMenuBar, QSpacerItem
from PyQt6.QtCore import Qt, QSize

class WasteClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waste Classifier")
        self.resize(900, 1000)

        self.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #fafafa;
            }
            QWidget {
                background-color: #151515;
            }
            QPushButton {
                font-size: 16px;
                color: #fafafa;
                border-radius: 6px;
                border: 1px solid white;
            }
        """)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.widgetCentral = QWidget()
        self.setCentralWidget(self.widgetCentral)
        self.widgetCentral.setLayout(self.layout)

        self.barMenu = QMenuBar()
        self.barMenu.setStyleSheet("border: 1px solid white;")
        self.barMenu.setFixedHeight(30)
        self.layout.addWidget(self.barMenu)

        self.containerMainLayout = QVBoxLayout()
        self.containerMainLayout.setContentsMargins(80, 80, 80, 80)
        self.containerMain = QWidget()
        self.layout.addWidget(self.containerMain)
        self.containerMain.setStyleSheet("border: 1px solid red;")
        self.containerMain.setLayout(self.containerMainLayout)

        self.containerImageButtonsLayout = QVBoxLayout()
        self.containerImageButtonsLayout.setContentsMargins(20, 20, 20, 20)
        self.containerImageButtonsLayout.setSpacing(0)
        self.containerImageButtons = QWidget()
        self.containerMainLayout.addWidget(self.containerImageButtons)
        self.containerImageButtons.setStyleSheet("border: 1px solid white;")
        self.containerImageButtons.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.containerImageButtons.setLayout(self.containerImageButtonsLayout)

        self.containerImageLayout = QHBoxLayout()
        self.containerImageLayout.addStretch()
        self.containerImage = QWidget()
        self.containerImageButtonsLayout.addWidget(self.containerImage)
        self.containerImage.setStyleSheet("border: 1px solid red;")
        self.containerImage.setLayout(self.containerImageLayout)

        self.labelImage = SquareLabel("No image uploaded")
        self.containerImageLayout.addWidget(self.labelImage)
        self.containerImageLayout.addStretch()
        self.labelImage.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.spacer = QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.containerImageButtonsLayout.addItem(self.spacer)

        self.buttonUpload = QPushButton("Upload Image")
        self.buttonClassify = QPushButton("Classify")
        self.buttonClear = QPushButton("Clear")
        self.buttonUpload.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.buttonClassify.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.buttonClear.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.containerButtonsLayout = QGridLayout()
        self.containerButtonsLayout.setContentsMargins(20, 20, 20, 20)
        self.containerButtonsLayout.addWidget(self.buttonUpload, 0, 0)
        self.containerButtonsLayout.addWidget(self.buttonClear, 0, 1)
        self.containerButtonsLayout.addWidget(self.buttonClassify, 1, 0, 1, 2)
        self.containerButtonsLayout.setColumnStretch(0, 1)
        self.containerButtonsLayout.setColumnStretch(1, 1)
        self.containerButtonsLayout.setRowStretch(0, 1)
        self.containerButtonsLayout.setRowStretch(1, 1)

        self.containerButtons = QWidget()
        self.containerImageButtonsLayout.addWidget(self.containerButtons)
        self.containerButtons.setStyleSheet("border: 1px solid white;")
        self.containerButtons.setMinimumHeight(150)
        self.containerButtons.setMaximumHeight(200)
        self.containerButtons.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.containerButtons.setLayout(self.containerButtonsLayout)

class SquareLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border-radius: 10px; border: 1px solid white")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(100, 100)
        self.setMaximumSize(500, 500)

    def hasHeightForWidth(self):
        return True
    
    def heightForWidth(self, width):
        return width

    def sizeHint(self):
        return QSize(500, 500)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WasteClassifierApp()
    window.show()
    app.exec()