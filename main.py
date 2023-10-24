import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, \
    QListWidget
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F


class ImageCropper(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        layout = QVBoxLayout()

        self.loadButton = QPushButton("Загрузить изображения", self)
        self.loadButton.clicked.connect(self.loadImages)
        layout.addWidget(self.loadButton)

        self.cropButton = QPushButton("Разделить на людей", self)
        self.cropButton.clicked.connect(self.cropImages)
        layout.addWidget(self.cropButton)

        self.imageList = QListWidget(self)
        layout.addWidget(self.imageList)
        self.imageList.itemDoubleClicked.connect(self.removeItem)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle("Обрезка изображений")
        self.setGeometry(100, 100, 800, 600)

    def removeItem(self):
        currentRow = self.imageList.currentRow()
        self.imageList.takeItem(currentRow)

    def loadImages(self):
        options = QFileDialog.Options()
        images, _ = QFileDialog.getOpenFileNames(self, "Загрузить изображения", "",
                                                 "Изображения (*.png *.jpg *.jpeg);;Все файлы (*)", options=options)
        if images:
            self.imageList.addItems(images)

    def cropImages(self):
        for index in range(self.imageList.count()):
            image_path = self.imageList.item(index).text()
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor_img = F.to_tensor(image_rgb).unsqueeze(0)
            img_name = os.path.basename(image_path).split('.')[0]

            with torch.no_grad():
                prediction = self.model(tensor_img)

            for i, box in enumerate(prediction[0]['boxes']):
                if prediction[0]['labels'][i] == 1 and prediction[0]['scores'][i] > 0.95:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_image = image[y1:y2, x1:x2]
                    output_dir = os.path.join(os.path.dirname(image_path), img_name)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{i}.jpg")
                    cv2.imwrite(output_path, cropped_image)
            print(f"Обрезано {len(prediction[0]['boxes'])} людей из {image_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageCropper()
    ex.show()
    sys.exit(app.exec_())
