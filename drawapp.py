from json import load
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class drawnumber(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = QImage(QSize(280, 280), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brush_size = 30
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.loaded_model = None
        self.load_default_model()
        self.initUI()

    def initUI(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('File')

        load_model_action = QAction('Load PyTorch Model', self)
        load_model_action.setShortcut('Ctrl+L')
        load_model_action.triggered.connect(self._open_file_dialog_and_load_model)
        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save)
        clear_action = QAction('Clear', self)
        clear_action.setShortcut('Ctrl+C')
        clear_action.triggered.connect(self.clear)

        filemenu.addAction(load_model_action)
        filemenu.addAction(save_action)
        filemenu.addAction(clear_action)

        self.statusbar = self.statusBar()
        self.setWindowTitle('MNIST Classifier (PyTorch)')
        self.setGeometry(300, 300, 350, 350)
        self.show()

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            scaled_image = self.image.scaled(28, 28)
            arr = np.zeros((28, 28))

            for i in range(28):
                for j in range(28):
                    arr[j, i] = 1.0 - scaled_image.pixelColor(i, j).getRgb()[0] / 255.0
            
            arr_tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
            
            if self.loaded_model:
                try:
                    self.loaded_model.eval()
                    with torch.no_grad():
                        output = self.loaded_model(arr_tensor)
                        probabilities = F.softmax(output, dim=1)
                        predict_num = str(torch.argmax(probabilities).item())
                        self.statusbar.showMessage(f'Predicted Number: {predict_num} (Confidence: {probabilities.max().item():.2f})')
                except Exception as e:
                    self.statusbar.showMessage(f'Error during prediction: {e}')
            else:
                self.statusbar.showMessage('Error: Model not loaded. Go to File -> Load PyTorch Model.')

    def _load_model_core(self, fname):
        model = MNIST_CNN()
        model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
        self.loaded_model = model
        return True

    def _open_file_dialog_and_load_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open PyTorch Model', '', 'PyTorch Model (*.pt *.pth);;All Files(*.*)')

        if fname:
            try:
                self._load_model_core(fname)
                self.statusbar.showMessage(f'PyTorch Model Loaded Successfully from: {fname}')
            except Exception as e:
                self.statusbar.showMessage(f'Error loading model: {e}. Check if model architecture matches MNIST_CNN.')
                self.loaded_model = None

    def load_default_model(self):
        default_path = "./data/3001_mnist.pt"
        try:
            self._load_model_core(default_path)
            self.statusbar.showMessage(f'Default PyTorch Model Loaded: {default_path}')
        except FileNotFoundError:
            self.statusbar.showMessage(f'Error: Default model not found at {default_path}. Please load manually via File menu.')
        except Exception as e:
            self.statusbar.showMessage(f'Error loading model: {e}. Check if model architecture matches MNIST_CNN.')
            self.loaded_model = None

    def save(self):
        fpath, _= QFileDialog.getSaveFileName(self, 'Save Image', '', 'PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*)')

        if fpath:
            self.image.scaled(28, 28).save(fpath)
            self.statusbar.showMessage(f'Image saved to {fpath}')

    def clear(self):
        self.image.fill(Qt.white)
        self.statusbar.clearMessage()
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = drawnumber()
    sys.exit(app.exec_())