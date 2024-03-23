import numpy as np
import cv2

def detector_face_image(prototxt_path,model_path,image_path):
    # 加载
    print("loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)

    #构造blob
    image = cv2.imread(image_path)
    image = cv2.resize(image,(300,300))
    (h,w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300),
                                 (104.0,177.0,123.0))

    # 检测人脸
    print("detecting face... ")
    net.setInput(blob)
    detections = net.forward()

    #遍历
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        default_confidence = 0.5
        # 过滤弱检测
        if confidence > default_confidence:
            # 获取检测框坐标
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 绘制框
            print("confidence： {:.3f}".format(confidence))
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if (startY - 10) > 10 else (startY +10)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image,text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0,0,255),1)

    cv2.imshow("Output_Image",image)
    cv2.waitKey(0)

if __name__=="__main__":
    # prototxt文件路径
    prototxt_path =r"d:\Opencv4.10.0\opencv\sources\samples\dnn\face_detector\deploy.prototxt"
    # model文件路径
    model_path = r"d:\Opencv4.10.0\opencv\sources\samples\dnn\face_detector\res10_300x300_ssd_iter_140000_fp16 (1).caffemodel"
    # image路径
    image_path = r"d:\img\face.jpg"

    detector_face_image(prototxt_path, model_path, image_path)
