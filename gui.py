import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenu, QAction, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QIcon, QPalette, QColor, QPixmap, QImage
import cv2
import torch
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
from f_utils import init_models, get_output_count, get_index_max
from privacy import Ui_PrivacyWindow
from instructions import Ui_InstructionsWindow
from about import Ui_AboutWindow


class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.setWindowTitle("Facemask Detection")
        self.setWindowIcon(QIcon('facemask_icon.png'))
        self.setMinimumSize(600, 400)
        self.resize(600, 400)

        self.closeFlag = False
        self.theme = "dark"
        self.language = "uk"

        # Use default webcam
        self._setWebcam(0)

        #self._createActions()
        self._createMenuBar()
        self._createLayout()
        self._connectActions()

        # Set default theme to dark when launching the app
        self._setDarkTheme()

    def _createMenuBar(self):
        self.menuBar = self.menuBar()
        # File menu
        self.optionsMenu = QMenu("&Options", self)
        self.menuBar.addMenu(self.optionsMenu)
        self.themeMenu = self.optionsMenu.addMenu("Theme")
        self.lightTheme = self.themeMenu.addAction("Light")
        self.darkTheme = self.themeMenu.addAction("Dark")
        self.webcamMenu = self.optionsMenu.addMenu("Webcam")
        self.webcams = 2 # list_ports() pour aller plus vite

        if self.webcams == 0:
            print("Error, no webcam detected! Please plug your webcam.\n")
        elif self.webcams == 1:
            self.webcam0 = self.webcamMenu.addAction("0")
        elif self.webcams == 2:
            self.webcam0 = self.webcamMenu.addAction("0")
            self.webcam1 = self.webcamMenu.addAction("1")
        elif self.webcams == 3:
            self.webcam0 = self.webcamMenu.addAction("0")
            self.webcam1 = self.webcamMenu.addAction("1")
            self.webcam2 = self.webcamMenu.addAction("2")

        # for index in range(self.webcams):
        #     self.webcamMenu.addAction(str(index))

        # Help menu
        self.helpMenu = self.menuBar.addMenu(QIcon(":help-content.svg"), "&Help")
        self.instructionsAction = QAction("&Instructions", self)
        self.aboutAction = QAction("&About", self)
        self.confidentialityAction = QAction("&Confidentiality", self)
        self.helpMenu.addAction(self.instructionsAction)
        self.helpMenu.addAction(self.aboutAction)
        self.helpMenu.addAction(self.confidentialityAction)


    def _connectActions(self):
        self.startWebcamButton.clicked.connect(self.main_facemask_detection)

        # Themes
        self.darkTheme.triggered.connect(self._setDarkTheme)
        self.lightTheme.triggered.connect(self._setLightTheme)

        # Webcams
        if self.webcams == 1:
            self.webcam0.triggered.connect(lambda: self._setWebcam(0))
        elif self.webcams == 2:
            self.webcam0.triggered.connect(lambda: self._setWebcam(0))
            self.webcam1.triggered.connect(lambda: self._setWebcam(1))
        elif self.webcams == 3:
            self.webcam0.triggered.connect(lambda: self._setWebcam(0))
            self.webcam1.triggered.connect(lambda: self._setWebcam(1))
            self.webcam2.triggered.connect(lambda: self._setWebcam(2))

        # Languages
        self.frButton.clicked.connect(lambda: self._setLanguage("fr"))
        self.ukButton.clicked.connect(lambda: self._setLanguage("uk"))
        self.esButton.clicked.connect(lambda: self._setLanguage("es"))

        # Other Windows
        self.confidentialityAction.triggered.connect(self._openPrivacy)
        self.instructionsAction.triggered.connect(self._openInstructions)
        self.aboutAction.triggered.connect(self._openAbout)

    def _openPrivacy(self):
        self.win = Ui_PrivacyWindow(theme=self.theme)
        self.win.show()

    def _openInstructions(self):
        self.win = Ui_InstructionsWindow(theme=self.theme)
        self.win.show()

    def _openAbout(self):
        self.win = Ui_AboutWindow(theme=self.theme)
        self.win.show()

    def _setLanguage(self, language):

        self.language = language
        if language == "fr":
            print("Language set to french!")

            self.startWebcamButton.setText("Commencer la détection")

            self.helpMenu.setTitle("Aide")
            self.instructionsAction.setText("Instructions")
            self.aboutAction.setText("À propos")
            self.confidentialityAction.setText("Confidentialité")

            self.optionsMenu.setTitle("Options")
            self.webcamMenu.setTitle("Caméra")
            self.themeMenu.setTitle("Thème")
            self.darkTheme.setText("Sombre")
            self.lightTheme.setText("Jour")

        elif language == "uk":
            print("Language set to english!")

            self.startWebcamButton.setText("Start webcam")

            self.helpMenu.setTitle("Help")
            self.instructionsAction.setText("Instructions")
            self.aboutAction.setText("About")
            self.confidentialityAction.setText("Confidentiality")

            self.optionsMenu.setTitle("Options")
            self.webcamMenu.setTitle("Webcam")
            self.themeMenu.setTitle("Theme")
            self.darkTheme.setText("Dark")
            self.lightTheme.setText("Light")

        elif language == "es":
            print("Language set to spanish!")

            self.startWebcamButton.setText("Inicio de la detección")

            self.helpMenu.setTitle("Ayuda")
            self.instructionsAction.setText("Instrucciones")
            self.aboutAction.setText("Acerca de")
            self.confidentialityAction.setText("Confidencialidad")

            self.optionsMenu.setTitle("Opciones")
            self.webcamMenu.setTitle("Camara")
            self.themeMenu.setTitle("Tema")
            self.darkTheme.setText("Oscuro")
            self.lightTheme.setText("Dia")

    def _setDarkTheme(self):
        print("Dark Theme set!")
        self.theme = "dark"
        self.widget.setStyleSheet("color: rgb(255,255,255); background-color: rgb(30,30,30);")

    def _setLightTheme(self):
        print("Light Theme set!")
        self.theme = "light"
        self.widget.setStyleSheet("color: rgb(0,0,0); background-color: rgb(255,255,255);")

    def _createTopFlagsWidget(self):
        self.topFlagsWidget = QWidget()
        layout = QHBoxLayout()

        self.frButton = QPushButton()
        icon = QIcon()
        icon.addPixmap(QPixmap("images/flag-france.png"), QIcon.Normal, QIcon.Off)
        self.frButton.setIcon(icon)
        layout.addWidget(self.frButton)

        self.ukButton = QPushButton()
        icon = QIcon()
        icon.addPixmap(QPixmap("images/flag-uk.png"), QIcon.Normal, QIcon.Off)
        self.ukButton.setIcon(icon)
        layout.addWidget(self.ukButton)

        self.esButton = QPushButton()
        icon = QIcon()
        icon.addPixmap(QPixmap("images/flag-spain.png"), QIcon.Normal, QIcon.Off)
        self.esButton.setIcon(icon)
        layout.addWidget(self.esButton)

        self.topFlagsWidget.setLayout(layout)

        return self.topFlagsWidget

    def _createTopMainWidget(self):
        self.topMainWidget = QWidget()
        layout = QVBoxLayout()
        self.topFlagsWidget = self._createTopFlagsWidget()
        layout.addWidget(self.topFlagsWidget)
        self.startWebcamButton = QPushButton("Start webcam")
        layout.addWidget(self.startWebcamButton)
        self.topMainWidget.setLayout(layout)
        return self.topMainWidget

    def _createMainWidget(self):
        self.mainWidget = QWidget()
        # Create a QVBoxLayout instance
        layout = QVBoxLayout()
        # Add widgets to the layout
        self.topMainWidget = self._createTopMainWidget()
        layout.addWidget(self.topMainWidget)
        self.webcamLabel = QLabel("")
        self.webcamLabel.setScaledContents(True) # la webcam prendra tout l'espace du QLabel
        #self.webcamLabel.setPixmap(QtGui.QPixmap("7.png"))
        layout.addWidget(self.webcamLabel)
        layout.addWidget(QLabel(""))
        # Set the layout on the application's window
        self.mainWidget.setLayout(layout)
        return self.mainWidget

    def _createLayout(self):

        self.layout = QHBoxLayout()
        self.layout.addWidget(QLabel(""))
        self.centralWidget = self._createMainWidget()
        self.layout.addWidget(self.centralWidget)
        self.layout.addWidget(QLabel(""))
        #self.button1.setStyleSheet("border: 20px rgb(20,20,20); background: rgb(60,63,65) ; min-width: 100px ; max-width: 500px ; height: 50px;")
        #self.layout.setContentsMargins(200, 0, 200, 0)
        self.setLayout(self.layout)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def _setWebcam(self, webcamId):
        print("Webcam", webcamId, "set!")
        self.webcam = webcamId
        self.cap = cv2.VideoCapture(self.webcam)

    def closeEvent(self, a0: QtGui.QCloseEvent):
        self.closeFlag = True
        print("Window closed!\nSee you soon.\n")

    def displayImage(self, img, imgWindow=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2] == 4):
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.webcamLabel.setPixmap(QPixmap.fromImage(img))
        self.webcamLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

    def main_facemask_detection(self):
        # ATTENTION : il est important de ne pas avoir d'espace dans le path absolu des fichiers sinon erreur avec readNet
        FRAME_THICKNESS = 2
        modelFile = os.getcwd() + os.sep + "models" + os.sep + "res10_300x300_ssd_iter_140000.caffemodel"
        configFile = os.getcwd() + os.sep + "models" + os.sep + "deploy-prototxt.txt"
        print(configFile)
        print(os.path.isfile(configFile))
        net = cv2.dnn.readNet(configFile, modelFile)

        GREEN = (0, 255, 0)
        ORANGE = (0, 165, 255)
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        num_classes = 3

        image_transforms = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

        resnet18 = models.resnet18(pretrained=True)
        vgg16 = models.vgg16(pretrained=True)
        densenet = models.densenet161(pretrained=True)
        googlenet = models.googlenet(pretrained=True)
        shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
        resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
        wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
        mnasnet = models.mnasnet1_0(pretrained=True)

        cnn_models = [resnet18, vgg16, densenet, googlenet, shufflenet, mobilenet_v3_large, resnext50_32x4d,
                      wide_resnet50_2, mnasnet]
        init_models(cnn_models, num_classes)

        resnet18.load_state_dict(torch.load(r"models\facemask1500\resnet18_10epochs_10bs.pth"))
        vgg16.load_state_dict(torch.load(r"models\facemask1500\vgg16_10epochs_10bs.pth"))
        densenet.load_state_dict(torch.load(r"models\facemask1500\densenet_10epochs_10bs.pth"))
        googlenet.load_state_dict(torch.load(r"models\facemask1500\googlenet_10epochs_10bs.pth"))
        mnasnet.load_state_dict(torch.load(r"models\facemask1500\mnasnet_10epochs_10bs.pth"))
        mobilenet_v3_large.load_state_dict(torch.load(r"models\facemask1500\mobilenet_v3_large_10epochs_10bs.pth"))
        resnext50_32x4d.load_state_dict(torch.load(r"models\facemask1500\resnext50_32x4d_10epochs_10bs.pth"))
        shufflenet.load_state_dict(torch.load(r"models\facemask1500\shufflenet_10epochs_10bs.pth"))
        wide_resnet50_2.load_state_dict(torch.load(r"models\facemask1500\wide_resnet50_2_10epochs_10bs.pth"))

        cnn_models = [vgg16, resnext50_32x4d, densenet]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model in cnn_models:
            model.eval()
            model = model.to(device)

        preds_words = ['Correct', 'Incorrect', 'NoMask']
        preds_color = [GREEN, ORANGE, RED]
        color = BLUE
        text = 'Undefined'
        fps = 0
        faces_preds_list = []
        last_faces_count = 0
        faces_change = False

        # Main Loop
        while True:
            # Verifier nombre d'images captées chaque seconde
            img = self.cap.read()[1]
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
            net.setInput(blob)
            faces = net.forward()
            faces_count = 0

            for j in range(faces.shape[2]):
                confidence = faces[0, 0, j, 2]
                if confidence > 0.5:
                    faces_count += 1

            if faces_count != last_faces_count:
                faces_change = True
                last_faces_count = faces_count

            faces_count = 0
            if fps == 15 or faces_change:
                fps = 0
                faces_change = False
                faces_preds_list = []

            for i in range(faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > 0.5:
                    # print("Tete n° :", faces_count)
                    faces_count += 1
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    # print(x, y, x1, y1)
                    preds_list = []

                    if fps == 0:
                        # DEBUT IF
                        img_face = img[y:y1, x:x1]
                        img_opencv = cv2.resize(img_face, (224, 224))
                        img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_opencv)
                        img_tensor = image_transforms(img_pil).unsqueeze_(0)
                        img_tensor = img_tensor.to(device)

                        for model in cnn_models:
                            output = model(img_tensor)
                            _, preds = torch.max(output, 1)
                            preds_list.append(preds[0].item())

                        output_avg, vgg16_flag = get_output_count(preds_list)
                        index_value = get_index_max(output_avg, vgg16_flag)
                        print(preds_list, index_value)
                        text = preds_words[index_value]
                        color = preds_color[index_value]
                        faces_preds_list.append([text, color])
                        # FIN IF

                    cv2.rectangle(img, (x, y), (x1, y1), faces_preds_list[faces_count - 1][1], FRAME_THICKNESS)
                    cv2.putText(img, faces_preds_list[faces_count - 1][0], (x + 10, y + 20), font, fontScale,
                                faces_preds_list[faces_count - 1][1], 2, cv2.LINE_AA)

            fps += 1
            self.displayImage(img, 1)
            # cv2.imshow('webcam', img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if self.closeFlag:
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
