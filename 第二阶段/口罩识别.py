import cv2
import numpy as np
from keras.models import load_model

# 加载预训练的口罩检测模型
model = load_model('demo\my_model.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 初始化平均置信度
average_confidence = 0.0
confidence_count = 0

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
while True:
    # 从摄像头读取一帧图像
    ret, frame = cap.read()

    # 如果读取失败，则退出循环
    if not ret:
        print("无法接收摄像头帧（流结束？）。退出...")
        break
    # 将图像转换为灰度图，因为Haar级联需要灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 在每个检测到的人脸上绘制矩形框
    for (x, y, w, h) in faces:
        # 裁剪人脸区域
        face = frame[y:y + h, x:x + w]
        # 对图像进行预处理，例如调整大小以适应模型输入
        # 假设模型接受224x224大小的RGB图像
        # 调整图像大小
        face = cv2.resize(face, (224, 224))
        # 转换图像颜色空间
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # 增加批次维度
        face = np.expand_dims(face, axis=0)
        # 增加通道维度
        # preprocessed_frame = np.expand_dims(preprocessed_frame, axis=-1)

        # 使用模型进行预测
        predictions = model.predict(face)
        print(predictions)
        print(predictions.shape)
        # 根据预测结果进行处理
        # 这里假设模型输出是一个概率值，大于0.5则认为戴了口罩
        if predictions[0][0] < 0.6:
            label = "Mask:"
            color = (0, 255, 0)
        else:
            label = "No Mask:"
            color = (0, 0, 255)

            # 更新平均置信度
        confidence = round(predictions[0][0])
        average_confidence += confidence
        confidence_count += 1
        # 计算标签的文本大小和位置
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = x + (w - text_size[0]) // 2  # 水平居中
        text_y = y - text_size[1] - 10  # 垂直位置在矩形框上方并稍有偏移
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if confidence_count > 0:
            average_confidence /= confidence_count
        accuracy_text = f"{average_confidence * 100:.0f}%"
        print(accuracy_text)
        # 显示识别准确率
        # accuracy_text = f" {accuracy * 100}%"
        # 在图像上添加标签
        cv2.putText(frame, label + accuracy_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 显示结果图像
    cv2.imshow('Mask Detection', frame)

    # 等待按键，如果按下'q'则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
