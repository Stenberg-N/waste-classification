import sys, os
from src.evaluate import predict_image
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QPushButton, QGridLayout, QMenuBar, QSpacerItem, QFileDialog, QMessageBox
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QColor, QPainter, QBrush, QAction

class WasteClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waste Classifier")
        self.resize(900, 1000)
        self.image_path = None

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

        self.barMenu = self.menuBar()
        self.barMenu.setStyleSheet("border: 1px solid white;")
        self.barMenu.setFixedHeight(30)
        self.layout.addWidget(self.barMenu)

        self.fileMenuBar = QMenuBar()
        self.fileMenu = self.fileMenuBar.addMenu("File")
        self.barMenu.setCornerWidget(self.fileMenuBar, Qt.Corner.TopLeftCorner)

        self.actionUpload = QAction("Upload image")
        self.actionClear = QAction("Clear image")
        self.actionClassify = QAction("Classify image")
        self.actionClassify.setEnabled(False)
        self.actionExit = QAction("Exit")

        self.actionUpload.triggered.connect(self.uploadImage)
        self.actionClear.triggered.connect(self.clearImage)
        self.actionClassify.triggered.connect(self.classifyImage)
        self.actionExit.triggered.connect(self.close)

        self.fileMenu.addAction(self.actionUpload)
        self.fileMenu.addAction(self.actionClear)
        self.fileMenu.addAction(self.actionClassify)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionExit)

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

        self.labelImage = SquareLabel("No image uploaded \n\n Drag and drop an image here \n\n")
        self.containerImageLayout.addWidget(self.labelImage)
        self.containerImageLayout.addStretch()
        self.labelImage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelImage.setScaledContents(True)
        self.labelImage.setAcceptDrops(True)
        self.labelImage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.containerFileResultsLayout = QVBoxLayout()
        self.containerFileResultsLayout.setContentsMargins(10, 10, 10, 10)
        self.containerFileResults = QWidget()
        self.containerImageButtonsLayout.addWidget(self.containerFileResults)
        self.containerFileResults.setStyleSheet("border: 1px solid cyan;")
        self.containerFileResults.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.containerFileResults.setLayout(self.containerFileResultsLayout)

        self.labelResults = QLabel("Prediction: ")
        self.containerFileResultsLayout.addWidget(self.labelResults)
        self.labelResults.setStyleSheet("border: 1px solid green;")
        self.labelResults.setMaximumHeight(30)

        self.labelConfidence = QLabel("Confidence: ")
        self.containerFileResultsLayout.addWidget(self.labelConfidence)
        self.labelConfidence.setStyleSheet("border: 1px solid green;")
        self.labelConfidence.setMaximumHeight(30)

        self.labelFile = QLabel("Uploaded image: ")
        self.containerFileResultsLayout.addWidget(self.labelFile)
        self.labelFile.setStyleSheet("border: 1px solid green;")
        self.labelFile.setMaximumHeight(30)

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
        self.containerButtons.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.containerButtons.setLayout(self.containerButtonsLayout)

        self.buttonUpload.clicked.connect(self.uploadImage)
        self.buttonClassify.clicked.connect(self.classifyImage)
        self.buttonClear.clicked.connect(self.clearImage)

    def makeRoundedPixmap(self, pixmap, radius=10):
        scaled = pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        rounded = QPixmap(scaled.size())
        rounded.fill(QColor("transparent"))
        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(scaled))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(scaled.rect(), radius, radius)
        painter.end()
        return rounded

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        url = event.mimeData().urls()[0]
        image_path = url.toLocalFile()
        if image_path.lower().endswith(('.png', '.jpg')):
            pixmap = QPixmap(url.toLocalFile())
            rounded_pixmap = self.makeRoundedPixmap(pixmap)
            self.labelImage.setPixmap(rounded_pixmap)
            self.labelFile.setText(f"Uploaded image: {url.fileName()}")
            self.image_path = image_path
        else:
            msg = QMessageBox(QMessageBox.Icon.Warning, "Invalid File", "Please drop an image file (.png, .jpg)", parent=self)
            msg.exec()

    def uploadImage(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg)"
        )
        if filename:
            self.image_path = filename
            self.labelFile.setText(f"Uploaded Image: {os.path.basename(filename)}")
            pixmap = QPixmap(filename)
            rounded_pixmap = self.makeRoundedPixmap(pixmap)
            self.labelImage.setPixmap(rounded_pixmap)
            self.checkImageUpload()

    def classifyImage(self):
        if not self.image_path:
            msg = QMessageBox(QMessageBox.Icon.Critical, "Error", "Please upload an image!", parent=self)
            msg.exec()
            return

        predicted_class, confidence = predict_image(self.image_path)
        self.labelResults.setText(f"Prediction: {predicted_class}")
        self.labelConfidence.setText(f"Confidence: {confidence * 100:.2f}% sure")

    def clearImage(self):
        self.image_path = None
        self.labelImage.clear()
        self.labelImage.setText("No image uploaded \n\n Drag and drop an image here \n\n")
        self.labelResults.setText("Prediction: ")
        self.labelConfidence.setText("Confidence: ")
        self.labelFile.setText("Uploaded image: ")
        self.checkImageUpload()

    def checkImageUpload(self):
        if self.image_path is not None:
            self.actionClassify.setEnabled(True)
        else:
            self.actionClassify.setEnabled(False)

class SquareLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: transparent;")
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