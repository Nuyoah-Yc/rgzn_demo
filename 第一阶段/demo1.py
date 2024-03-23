import cv2

# 加载Haar特征分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
img = cv2.imread('demo1.png')

# 转换为灰度图，因为Haar分类器需要灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测图像中的人脸
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# 在检测到的人脸周围画矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 显示带有人脸识别框的图像
cv2.imshow('img', img)

# 等待按键后退出
cv2.waitKey()

