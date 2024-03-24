# face_detector_sdk/face_detector_sdk/detector.py
import numpy as np
import cv2


def detector_face_image():
    print("Loading model...")
    net = cv2.dnn.readNetFromCaffe('./deploy.prototxt', './res10_300x300_ssd_iter_140000_fp16 (1).caffemodel')

    image = cv2.imread('./demo1.png')
    image = cv2.resize(image, (300, 300))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    print("Detecting face...")
    net.setInput(blob)
    detections = net.forward()

    faces = []
    default_confidence = 0.5
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > default_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            faces.append(box.astype("int"))

    return faces
