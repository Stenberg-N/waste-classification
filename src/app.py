import sys, os
from src.evaluate import predict_image
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QPushButton, QGridLayout, QMenuBar, QSpacerItem, QFileDialog, QMessageBox, QDialog, QMenu
from PyQt6.QtCore import Qt, QSize, QSettings, pyqtSignal
from PyQt6.QtGui import QPixmap, QColor, QPainter, QBrush, QAction

class WasteClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waste Classifier")
        self.resize(900, 1000)
        self.image_path = None
        self.settings = QSettings("Stenberg-N", "WasteClassifierApp")
        self.currentTheme = self.settings.value("theme", "light")

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.widgetCentral = QWidget()
        self.setCentralWidget(self.widgetCentral)
        self.widgetCentral.setLayout(self.layout)

        self.barMenu = self.menuBar()
        self.barMenu.setFixedHeight(40)

        self.containerBarMenuOptionsLayout = QHBoxLayout()
        self.containerBarMenuOptions = QWidget()
        self.containerBarMenuOptionsLayout.setContentsMargins(0, 0, 0, 0)
        self.containerBarMenuOptionsLayout.setSpacing(0)
        self.containerBarMenuOptions.setLayout(self.containerBarMenuOptionsLayout)

        self.fileMenuBar = CustomMenuBar()
        self.fileMenu = CustomMenu("File")
        self.fileMenuBar.addMenu(self.fileMenu)

        self.helpMenuBar = CustomMenuBar()
        self.helpMenu = CustomMenu("Help")
        self.helpMenuBar.addMenu(self.helpMenu)

        self.containerBarMenuOptionsLayout.addWidget(self.fileMenuBar)
        self.containerBarMenuOptionsLayout.addWidget(self.helpMenuBar)
        self.barMenu.setCornerWidget(self.containerBarMenuOptions, Qt.Corner.TopLeftCorner)

        self.buttonToggleTheme = HoverButton("Dark Theme" if self.currentTheme == "light" else "Light Theme")
        self.buttonToggleTheme.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.buttonToggleTheme.setMaximumWidth(150)
        self.buttonToggleTheme.setCheckable(True)
        self.buttonToggleTheme.clicked.connect(self.toggleTheme)
        self.barMenu.setCornerWidget(self.buttonToggleTheme, Qt.Corner.TopRightCorner)

        self.actionUpload = QAction("Upload image")
        self.actionClear = QAction("Clear image")
        self.actionClassify = QAction("Classify image")
        self.actionClassify.setEnabled(False)
        self.actionExit = QAction("Exit")
        self.actionImageUploading = QAction("Image upload")
        self.actionWasteInfo = QAction("Waste info")

        self.actionUpload.triggered.connect(self.uploadImage)
        self.actionClear.triggered.connect(self.clearImage)
        self.actionClassify.triggered.connect(self.classifyImage)
        self.actionExit.triggered.connect(self.close)
        self.actionImageUploading.triggered.connect(self.dialogImageUploading)
        self.actionWasteInfo.triggered.connect(self.dialogWasteInfo)

        self.fileMenu.addAction(self.actionUpload)
        self.fileMenu.addAction(self.actionClear)
        self.fileMenu.addAction(self.actionClassify)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionExit)

        self.helpMenu.addAction(self.actionImageUploading)
        self.helpMenu.addAction(self.actionWasteInfo)

        self.containerMainLayout = QVBoxLayout()
        self.containerMainLayout.setContentsMargins(80, 80, 80, 80)
        self.containerMainLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.containerMain = QWidget()
        self.layout.addWidget(self.containerMain)
        self.containerMain.setLayout(self.containerMainLayout)

        self.containerImageButtonsLayout = QVBoxLayout()
        self.containerImageButtonsLayout.setContentsMargins(40, 40, 40, 40)
        self.containerImageButtonsLayout.setSpacing(10)
        self.containerImageButtons = QWidget()
        self.containerMainLayout.addWidget(self.containerImageButtons)
        self.containerImageButtons.setMaximumWidth(1000)
        self.containerImageButtons.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.containerImageButtons.setLayout(self.containerImageButtonsLayout)

        self.containerImageLayout = QHBoxLayout()
        self.containerImageLayout.addStretch()
        self.containerImage = QWidget()
        self.containerImageButtonsLayout.addWidget(self.containerImage)
        self.containerImage.setStyleSheet("border: 1px solid red;")
        self.containerImage.setLayout(self.containerImageLayout)

        self.labelImage = SquareLabel("""
            <p style='text-align: center;'>
                No image uploaded<br>
                <br>Drag and drop an image here,
                <br>or click me to open the file explorer
            </p>
        """)
        self.labelImage.setWordWrap(True)

        self.labelImage.clicked.connect(self.uploadImage)

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
        self.labelResults.setMaximumHeight(40)

        self.labelConfidence = QLabel("Confidence: ")
        self.containerFileResultsLayout.addWidget(self.labelConfidence)
        self.labelConfidence.setMaximumHeight(40)

        self.labelFile = QLabel("Uploaded image: ")
        self.containerFileResultsLayout.addWidget(self.labelFile)
        self.labelFile.setMaximumHeight(40)

        self.spacer = QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.containerImageButtonsLayout.addItem(self.spacer)

        self.buttonUpload = HoverButton("Upload Image")
        self.buttonClassify = HoverButton("Classify")
        self.buttonClear = HoverButton("Clear")
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
        self.containerButtons.setMinimumHeight(180)
        self.containerButtons.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.containerButtons.setLayout(self.containerButtonsLayout)

        self.buttonUpload.clicked.connect(self.uploadImage)
        self.buttonClassify.clicked.connect(self.classifyImage)
        self.buttonClear.clicked.connect(self.clearImage)


        # STYLING

        self.barMenu.setStyleSheet("""
            color: #fafafa;
            font-size: 16px;
            border: none;
            background-color: rgb(20, 20, 20);
            text-align: center;
        """)

        self.buttonToggleTheme.setStyleSheet("""
            margin: 8px 20px 8px 0;
        """)

        self.containerMain.setStyleSheet("""
            border-top: 2px solid #63e6ab;
        """)

        self.containerBarMenuOptions.setStyleSheet("""
            color: #fafafa;
            background-color: rgb(20, 20, 20);
        """)

        self.lightTheme = """
            QLabel {
                font-size: 16px;
                color: black;
            }
            QWidget {
                background-color: rgb(208, 193, 180);
                font-size: 16px;
            }
            QPushButton {
                font-size: 16px;
                color: black;
            }
            QMenu {
                background-color: rgb(20, 20, 20);
                color: #fafafa;
                border: 1px solid #63e6ab;
            }
            QMenu::item:selected {
                background-color: rgb(40, 40, 40);
            }
            QMenu::item:disabled {
                color: rgb(160, 160, 160);
            }
            QMenu:separator {
                height: 2px;
                background: #fafafa;
            }
        """

        self.darkTheme = """
            QLabel {
                font-size: 16px;
                color: #fafafa;
            }
            QWidget {
                background-color: rgb(20, 20, 20);
                font-size: 16px;
            }
            QPushButton {
                font-size: 16px;
                color: #fafafa;
            }
            QMenu {
                background-color: rgb(20, 20, 20);
                color: #fafafa;
                border: 1px solid #63e6ab;
            }
            QMenu::item:selected {
                background-color: rgb(40, 40, 40);
            }
            QMenu::item:disabled {
                color: rgb(160, 160, 160);
            }
            QMenu:separator {
                height: 2px;
                background: #fafafa;
            }
        """

        self.lightContainerImageButtons = """
            background-color: rgb(215, 215, 215);
            border-top: none;
            border: 1px solid rgb(110, 110, 110);
            border-radius: 10px;
            color: black;
        """

        self.lightContainerImage = """
            background-color: rgb(215, 215, 215);
            border: 2px solid rgb(110, 110, 110);
        """

        self.lightContainerFileResults = """
            border: 2px solid rgb(110, 110, 110);
            border-radius: 0;
        """

        self.lightContainerButtons = """
            border: 2px solid rgb(110, 110, 110);
            border-radius: 0;
        """

        self.lightButtonUpload = """
            QPushButton:hover {
                background-color: #52bf8e;
            }
            QPushButton {
                background-color: rgb(215, 215, 215);
                border: 2px solid rgb(110, 110, 110);
                border-radius: 6px;
            }
        """

        self.lightButtonClear = """
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 64);
            }
            QPushButton {
                background-color: rgb(215, 215, 215);
                border: 2px solid rgb(110, 110, 110);
                border-radius: 6px;
            }
        """

        self.lightButtonClassify = """
            QPushButton {
                background-color: #63e6ab;
                border: 3px solid #47a67b;
                border-radius: 6px;
                color: black;
            }
            QPushButton:hover {
                background-color: #52bf8e;
            }
        """

        self.lightLabelResults = """
            background-color: rgb(215, 215, 215);
            border: 2px solid rgb(110, 110, 110);
            border-radius: 0;
            padding: 5px 0 5px 5px;
        """

        self.lightLabelConfidence = """
            background-color: rgb(215, 215, 215);
            border: 2px solid rgb(110, 110, 110);
            border-radius: 0;
            padding: 5px 0 5px 5px;
        """

        self.lightLabelFile = """
            background-color: rgb(215, 215, 215);
            border: 2px solid rgb(110, 110, 110);
            border-radius: 0;
            padding: 5px 0 5px 5px;
        """

        self.lightLabelImage = """
            border: 2px solid rgb(180, 180, 180);
            border-radius: 10px;
            border-style: dashed; 
        """

        self.darkContainerImageButtons = """
            background-color: rgb(35, 35, 35);
            border-top: none;
            border: 1px solid rgb(72, 72, 72);
            border-radius: 10px;
        """

        self.darkContainerImage = """
            border: 2px solid rgb(128, 128, 128);
        """

        self.darkContainerFileResults = """
            border: 2px solid rgb(128, 128, 128);
            border-radius: 0;
        """

        self.darkContainerButtons = """
            background-color: rgb(35, 35, 35);
            border: 2px solid rgb(128, 128, 128);
            border-radius: 0;
        """

        self.darkButtonUpload = """
            QPushButton:hover {
                background-color: #52bf8e;
            }
            QPushButton {
                background-color: rgb(35, 35, 35);
                border: 2px solid rgb(128, 128, 128);
                border-radius: 6px;
            }
        """

        self.darkButtonClear = """
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 64);
            }
            QPushButton {
                background-color: rgb(35, 35, 35);
                border: 2px solid rgb(128, 128, 128);
                border-radius: 6px;
            }
        """

        self.darkButtonClassify = """
            QPushButton {
                background-color: #63e6ab;
                border: 3px solid #47a67b;
                border-radius: 6px;
                color: black;
            }
            QPushButton:hover {
                background-color: #52bf8e;
            }
        """

        self.darkLabelResults = """
            background-color: rgb(35, 35, 35);
            border: 2px solid rgb(128, 128, 128);
            border-radius: 0;
            padding: 5px 0 5px 5px;
        """

        self.darkLabelConfidence = """
            background-color: rgb(35, 35, 35);
            border: 2px solid rgb(128, 128, 128);
            border-radius: 0;
            padding: 5px 0 5px 5px;
        """

        self.darkLabelFile = """
            background-color: rgb(35, 35, 35);
            border: 2px solid rgb(128, 128, 128);
            border-radius: 0;
            padding: 5px 0 5px 5px;
        """

        self.darkLabelImage = """
            border: 2px solid rgb(60, 60, 60);
            border-radius: 10px;
            border-style: dashed;
        """

        self.applyTheme(self.currentTheme)

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
            self.msg = QMessageBox(QMessageBox.Icon.Critical, "Error", "Please upload an image!")
            self.msg.exec()
            return

        self.msg.setStyleSheet("""
            QPushButton {
                width: 20px;
            }
        """)

        predicted_class, confidence = predict_image(self.image_path)
        self.labelResults.setText(f"Prediction: {predicted_class}")
        self.labelConfidence.setText(f"Confidence: {confidence * 100:.2f}% sure")

    def clearImage(self):
        self.image_path = None
        self.labelImage.clear()
        self.labelImage.setText("""
            <p style='text-align: center;'>
                No image uploaded<br>
                <br>Drag and drop an image here
            </p>
        """)
        self.labelResults.setText("Prediction: ")
        self.labelConfidence.setText("Confidence: ")
        self.labelFile.setText("Uploaded image: ")
        self.checkImageUpload()

    def checkImageUpload(self):
        if self.image_path is not None:
            self.actionClassify.setEnabled(True)
        else:
            self.actionClassify.setEnabled(False)

    def dialogImageUploading(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle("Getting better results")
        self.dialog.resize(600, 450)

        self.header = QLabel("Tips to get better results from the model")
        self.header.setStyleSheet("font-size: 24px; color: #fafafa; font-weight: bold;")
        self.header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.label = QLabel("""
            <p style='text-align: left;'>
                If you are not getting the best prediction results, you can try a couple of things to fix that:
            </p>
            <div style='text-align: left; margin: 0 0 0 25px;'>
                • Make sure your picture is not blurry<br>
                • Try to take the picture from other angles<br>
                • Keep the background simple, like a white paper<br>
                • The model is only capable of identifying glass, metal, cardboard, paper, plastic and trash
            </div>
        """)
        self.label.setStyleSheet("font-size: 16px; color: #fafafa;")
        self.label.setWordWrap(True)

        self.buttonClose = HoverButton("Close")
        self.buttonClose.clicked.connect(self.dialog.close)
        self.buttonClose.setStyleSheet("""
            QPushButton {
                border: 1px solid #63e6ab;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgb(40, 40, 40);
            }
        """)

        self.dialogLayout = QVBoxLayout()
        self.dialogLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dialogLayout.setContentsMargins(20, 20, 20, 20)
        self.dialogLayout.setSpacing(0)
        self.dialogLayout.addWidget(self.header)
        self.dialogLayout.addSpacing(30)
        self.dialogLayout.addWidget(self.label)
        self.dialogLayout.addStretch()
        self.dialogLayout.addWidget(self.buttonClose)
        self.dialog.setLayout(self.dialogLayout)

        self.dialog.exec()

    def applyTheme(self, theme):
        app = QApplication.instance()
        if theme == "dark":
            app.setStyleSheet(self.darkTheme)
            self.containerImageButtons.setStyleSheet(self.darkContainerImageButtons)
            self.containerButtons.setStyleSheet(self.darkContainerButtons)
            self.labelResults.setStyleSheet(self.darkLabelResults)
            self.labelConfidence.setStyleSheet(self.darkLabelConfidence)
            self.labelFile.setStyleSheet(self.darkLabelFile)
            self.buttonUpload.setStyleSheet(self.darkButtonUpload)
            self.buttonClear.setStyleSheet(self.darkButtonClear)
            self.buttonClassify.setStyleSheet(self.darkButtonClassify)
            self.containerImage.setStyleSheet(self.darkContainerImage)
            self.containerFileResults.setStyleSheet(self.darkContainerFileResults)
            self.labelImage.setStyleSheet(self.darkLabelImage)
        else:
            app.setStyleSheet(self.lightTheme)
            self.containerImageButtons.setStyleSheet(self.lightContainerImageButtons)
            self.containerButtons.setStyleSheet(self.lightContainerButtons)
            self.labelResults.setStyleSheet(self.lightLabelResults)
            self.labelConfidence.setStyleSheet(self.lightLabelConfidence)
            self.labelFile.setStyleSheet(self.lightLabelFile)
            self.buttonUpload.setStyleSheet(self.lightButtonUpload)
            self.buttonClear.setStyleSheet(self.lightButtonClear)
            self.buttonClassify.setStyleSheet(self.lightButtonClassify)
            self.containerImage.setStyleSheet(self.lightContainerImage)
            self.containerFileResults.setStyleSheet(self.lightContainerFileResults)
            self.labelImage.setStyleSheet(self.lightLabelImage)

    def toggleTheme(self):
        self.currentTheme = "dark" if self.currentTheme == "light" else "light"
        self.applyTheme(self.currentTheme)
        self.buttonToggleTheme.setText("Dark Theme" if self.currentTheme == "light" else "Light Theme")
        self.settings.setValue("theme", self.currentTheme)

    def dialogWasteInfo(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle("Waste Category Information Page")
        self.dialog.resize(1100, 900)

        self.NavBarLayout = QHBoxLayout()
        self.NavBarLayout.setContentsMargins(20, 0, 0, 1)
        self.NavBarLayout.setSpacing(20)
        self.NavBarLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.NavBar = QMenuBar()
        self.NavBar.setFixedHeight(40)
        self.NavBar.setLayout(self.NavBarLayout)
        self.NavBar.setStyleSheet("""
            QMenuBar {
                border-bottom: 2px solid #63e6ab;
            }
        """)

        categories = ["Plastic", "Metal", "Glass", "Cardboard", "Paper"]

        for cat in categories:
            button = HoverButton(cat)
            self.NavBarLayout.addWidget(button)
            button.setMaximumWidth(85)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            button.clicked.connect(getattr(self, f"changeTo{cat}"))

            color = '#63e6ab'
            button.setStyleSheet(f"""
                QPushButton {{
                    border: 1px solid {color};
                }}
                QPushButton:hover {{
                background-color: rgb(40, 40, 40);
                }}
            """)

            setattr(self, f"button{cat}", button)

        self.containerInfoLayout = QVBoxLayout()
        self.containerInfo = QLabel("""
            <p style='text-align: left; font-size: 24px; font-weight: bold; color: #fafafa; margin: 0 0 20px 20px;'>
                Main Screen
            </p>
            <p style='text-align: left; font-size: 16px; color: #fafafa; margin: 0 20px 0 20px;'>
                This is the help page for information on different waste.<br><br>
                On the top you can see a bar with all the different trash types. Each tab has information on that specific trash type e.g.
                what sub-materials are there, what it is used for, its recyclability and random notes.
            </p>
        """)
        self.containerInfo.setWordWrap(True)
        self.containerInfo.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.containerInfo.setLayout(self.containerInfoLayout)

        self.buttonClose = HoverButton("Close")
        self.buttonClose.clicked.connect(self.dialog.close)
        self.buttonClose.setStyleSheet("""
            QPushButton {
                border: 1px solid #63e6ab;
                border-radius: 8px;
                margin: 0 20px 20px 20px;
            }
            QPushButton:hover {
                background-color: rgb(40, 40, 40);
            }
        """)

        self.dialogLayout = QVBoxLayout()
        self.dialogLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dialogLayout.setContentsMargins(0, 0, 0, 0)
        self.dialogLayout.setSpacing(0)
        self.dialogLayout.addWidget(self.NavBar)
        self.dialogLayout.addSpacing(20)
        self.dialogLayout.addWidget(self.containerInfo)
        self.dialogLayout.addStretch()
        self.dialogLayout.addWidget(self.buttonClose)
        self.dialog.setLayout(self.dialogLayout)

        self.dialog.exec()

    def changeToPlastic(self):
        self.containerInfo.setText("""
            <p style='text-align: left; font-size: 24px; font-weight: bold; color: #fafafa; margin: 0 0 20px 20px;'>
                About Plastic
            </p>
            <p style='text-align: left; font-size: 16px; color: #fafafa; margin: 0 20px 0 20px; line-height: 1.5;'>
                Different plastics have differenet capabilities and common uses. Many products have a resin code (1-7) that indicates the plastic type.<br><br>
                Common plastic types are:<br>
                1 | PET/PETE: Polyethylene, Terephthalate. Used in beverage bottles and food containers. Highly recyclable. Safe for food.<br>
                2 | HDPE: High-Density Polyethylene. Used in milk jugs, shampoo bottles. Highly recyclable. Safe for food.<br>
                3 | PVC: Polyvinyl Chloride. Used in pipes, vinyl flooring. Rarely recyclable. Contains additives. Sensitive to heat.<br>
                4 | LDPE: Low-Density Polyethylene. Used in plastic bags, wraps and squeezable bottles. Sometimes recyclable.<br>
                5 | PP: Polypropylene. Used in food containers, yogurt cups, bottle caps. Increasingly recyclable.<br>
                6 | PS: Polystyrene(Styrofoam). Used in take-out boxes, cups, packing. Not easily recyclable. Breaks into microplastics.<br>
                7 | Other: Mixed plastics including PLA. Used in multi-layer packaging. Recyclability varies.
            </p>
        """)

    def changeToMetal(self):
        self.containerInfo.setText("""
            <p style='text-align: left; font-size: 24px; font-weight: bold; color: #fafafa; margin: 0 0 20px 20px;'>
                About Metal
            </p>
            <p style='text-align: left; font-size: 16px; color: #fafafa; margin: 0 20px 0 20px; line-height: 1.5;'>
                Metals like aluminum and steel can be recycled endlessly without loss of quality. Steel and aluminum can be properly sorted
                by the use of magnets.<br><br>
                Common metal types are:<br>
                Aluminum: Used in beverage cans, foil and food trays. Highly recyclable.<br>
                Steel: Used in food cans, aerosol cans. Highly recyclable. Magnetic.<br>
                Tin-plated steel: Used in tin cans, cookie tins. Recyclable. Not actually tin.<br>
                Copper: Used in wires and pipes. Highly recyclable. Valuable.<br>
                Brass: Used in fixtures and keys. Recyclable.
            </p>
        """)

    def changeToGlass(self):
        self.containerInfo.setText("""
            <p style='text-align: left; font-size: 24px; font-weight: bold; color: #fafafa; margin: 0 0 20px 20px;'>
                About Glass
            </p>
            <p style='text-align: left; font-size: 16px; color: #fafafa; margin: 0 20px 0 20px; line-height: 1.5;'>
                Like metals, glass is highly recyclable and can be reprocessed indefinitely. In terms of recycling, only container glass is
                accepted. Items like ceramics, window glass and Pyrex contaminate recycling streams.<br><br>
                Common glass types are:<br>
                Container glass: Used in containing food and liquid. Highly recyclable.<br>
                Clear glass: Water bottles and jars. Recyclable.<br>
                Green/Brown glass: Used in beer and wine bottles. Recyclable. The glass' color is to protect the contents from light, mainly UV.
                UV breaks down the compounds in beer and wine.<br>
                Tempered glass: Used in oven and shower doors. Not recyclable in normal bins. Has a different melting point.<br>
                Ceramic/Porcelain: Used in mugs and dishes. Not recyclable. Contains a crystalline structure.<br>
                Pyrex/Borosilicate: Used in bakeware. Not recyclable. Heat resistant glass.<br>
                Window glass: Used in windows and mirrors. Not recyclable. Contains coatings and additives.
            </p>
        """)

    def changeToCardboard(self):
        self.containerInfo.setText("""
            <p style='text-align: left; font-size: 24px; font-weight: bold; color: #fafafa; margin: 0 0 20px 20px;'>
                About Cardboard
            </p>
            <p style='text-align: left; font-size: 16px; color: #fafafa; margin: 0 20px 0 20px; line-height: 1.5;'>
                Cardboard includes corrugated shipping boxes and thinner paperboard packaging.<br><br>
                Common types of cardboard:<br>
                Corrugated cardboard: Used in shipping boxes and packaging. Highly recyclable. Strongest cardboard.<br>
                Paperboard/Cartonboard: Used in cereal and shoe boxes. Recyclable.<br>
                Greyboard: Used in notebook backs and packaging inserts. Recyclable. Has a slightly rough texture.<br>
                Wax-coated cardboard: Used in produce boxes like fruits and vegetables. Not recyclable. Water resistant.<br>
                Food-soiled cardboard: Used in pizza boxes and greasy containers. Partially recyclable. Greasy parts get composted or trashed.
            </p>
        """)

    def changeToPaper(self):
        self.containerInfo.setText("""
            <p style='text-align: left; font-size: 24px; font-weight: bold; color: #fafafa; margin: 0 0 20px 20px;'>
                About Paper
            </p>
            <p style='text-align: left; font-size: 16px; color: #fafafa; margin: 0 20px 0 20px; line-height: 1.5;'>
                Most clean paper, like office paper, newspaper, magazines and kraft bags, is recyclable. Contaminated items like tissues,
                thermal receipts and coated papers cannot be processed.<br><br>
                Common paper types:<br>
                Office/Printer paper: Used for documents and school paper. Highly recyclable.<br>
                Newspaper: Used in newspaper and newsprint ads. Recyclable. Often used for molded pulp.<br>
                Magazine/Glossy paper: Used in magazines and catalogs. Recyclable. The glossy coating is water-soluble.<br>
                Paper bags/Kraft paper: Used in grocery bags and packaging. Recyclable. Has strong fibers and can be composted.<br>
                Cardstock: Used in greeting cards and covers. Recyclable.<br>
                Tissue/Napkins: Used in tissues and paper towels. Not recyclable. Fibers are too short.<br>
                Thermal paper: Used in receipts and some tickets. Not recyclable. Contains chemicals like BPA/BPS.<br>
                Wax-coated/laminated paper: Used in some cups and fast-food wrappers. Not recyclable. Plastic/wax prevents pulping.
            </p>
        """)

class SquareLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(100, 100)
        self.setMaximumSize(500, 500)

    def hasHeightForWidth(self):
        return True
    
    def heightForWidth(self, width):
        return width

    def sizeHint(self):
        return QSize(500, 500)
    
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)

class HoverButton(QPushButton):
    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)

class CustomMenuBar(QMenuBar):
    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)

class CustomMenu(QMenu):
    def mouseMoveEvent(self, event):
        action = self.actionAt(event.pos())
        if action and action.isEnabled():
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WasteClassifierApp()
    window.show()
    app.exec()