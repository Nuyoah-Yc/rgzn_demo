import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
class faceRecognize(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('人脸检测界面')
        self.setFixedSize(1000, 800)

        self.image_name_label = QLabel(self)
        self.image_name_label.setGeometry(0, 0, 1000, 80)
        self.image_name_label.setAlignment(Qt.AlignCenter)
        self.image_name_label.setStyleSheet("background:#567fff")

        self.image_label = QLabel(self)
        self.image_label.setGeometry(150, 150, 550, 550)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border:2px solid gray")

        load_up_button = QPushButton('上传图像',self)
        load_up_button.setGeometry(800, 180, 90, 60)
        load_up_button.setStyleSheet("border-radius:10px;color:white;background:#567fff;font-size:15px")

        recognize_button = QPushButton('开始识别',self)
        recognize_button.setGeometry(800, 300, 90, 60)
        recognize_button.setStyleSheet("border-radius:10px;color:white;background:#567fff;font-size:15px")

        exit_button = QPushButton('退出',self)
        exit_button.setGeometry(800, 600, 90, 60)
        exit_button.setStyleSheet("border-radius:10px;color:white;background:gray;font-size:15px")

        load_up_button.clicked.connect(self.upload_image)
        recognize_button.clicked.connect(self.recognize)
        exit_button.clicked.connect(self.close)

    def upload_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Image(*.png *.jpg *.jpeg)")
        if file:
            image = QImage(file)
            if not image.isNull():
                # scaled_image 将是 QImage 对象尺寸与 image_label 的大小相匹配
                scaled_image = image.scaled(self.image_label.size(), Qt.KeepAspectRatio)
                # 调用了QPixmap类的fromImage静态方法，将scaled_image转换为QPixmap对象
                pixmap =QPixmap.fromImage(scaled_image)
                # 将QPixmap 对象（pixmap）设置为 QLabel 对象的图像内容
                self.image_label.setPixmap(pixmap)
                filename = self.image_name_label.setText(file)
                self.image = cv2.imread(file)
                if image is None:
                    print(f"Error: 图像文件 {filename} 无法加载。请检查文件路径和文件完整性。")
                    return
    def recognize(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x, y, w, h) in faces:
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        # img_rgb.data这是NumPy数组的数据缓冲区，包含了图像的实际像素数据
        q_image = QImage(img_rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = faceRecognize()
    w.show()
    sys.exit(app.exec_())

