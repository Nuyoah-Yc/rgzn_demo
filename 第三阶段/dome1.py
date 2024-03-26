import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from PyQt5.QtCore import Qt
import os


class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('人脸检测界面')
        self.setFixedSize(1000, 800)

        container = QHBoxLayout()

        self.image_name_label = QLabel(self)
        self.image_name_label.setAlignment(Qt.AlignCenter)
        self.image_name_label.setFixedSize(1000, 60)
        self.image_name_label.setStyleSheet("background: #567fff;")

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(550, 550)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        container.addWidget(self.image_label)

        button_layout = QVBoxLayout()
        upload_button = QPushButton('上传图像', self)
        recognize_button = QPushButton('开始识别', self)
        exit_button = QPushButton('退出', self)

        upload_button.setStyleSheet("background: #567fff;;border-radius:10px;color:white;font-size:18px")
        recognize_button.setStyleSheet("background: #567fff;;border-radius:10px;color:white;font-size:18px")
        exit_button.setStyleSheet("background:gray;border-radius:10px;color:white;font-size:18px")

        upload_button.setFixedSize(90, 60)
        recognize_button.setFixedSize(90, 60)
        exit_button.setFixedSize(90, 60)

        button_layout.addWidget(upload_button)
        button_layout.addWidget(recognize_button)
        button_layout.addWidget(exit_button)

        container.addLayout(button_layout)
        self.setLayout(container)

        upload_button.clicked.connect(self.upload_image)
        recognize_button.clicked.connect(self.recognize_face)
        exit_button.clicked.connect(self.close)

    def upload_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Image(*.png *.jpg *.jpeg)")
        if file:
            image = QImage(file)
            if not image.isNull():
                scaled_image = image.scaled(self.image_label.size(), Qt.KeepAspectRatio)
                pixmap = QPixmap.fromImage(scaled_image)
                self.image_label.setPixmap(pixmap)
                filename = os.path.basename(file)  # 获取文件名
                self.image_name_label.setText(filename)
                self.image_cv = cv2.imread(file)
                if self.image_cv is None:
                    print(f"Error: 图像文件 {filename} 无法加载。请检查文件路径和文件完整性。")
                    return

    def recognize_face(self):
        if self.image_cv is not None:
            gray = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2GRAY)

            prototxt_path = "./rlsb_sdk/deploy.prototxt"
            model_path = "./rlsb_sdk/res10_300x300_ssd_iter_140000_fp16 (1).caffemodel"
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

            (h, w) = gray.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(self.image_cv, (300, 300)), 1.0, (300, 300), (104, 177, 123))

            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    cv2.rectangle(self.image_cv, (startX, startY), (endX, endY), (0, 255.0), 2)
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(self.image_cv, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img_rgb = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            q_image = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())



