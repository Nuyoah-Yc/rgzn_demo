import cv2
# 设置帧的宽度和高度
frameWidth = 640
frameHeight = 480
# 加载车牌检测的Haar级联分类器
nPlateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
# 设置最小区域，用于过滤小的检测结果
minArea = 200
# 设置标记颜色（紫色
color = (255,0,255)
# 打开视频文件
cap = cv2.VideoCapture(r"img\car.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
# 设置亮度
cap.set(10,150)
# 初始化计数器
count = 0

while True:
    # 读取一帧图像
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用Haar级联分类器检测车牌(scaleFactor=1.1: 这是每次图像缩小的比例.minNeighbors=10: 这是一个对象周围被认为是真正的对象的邻居数。)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area >minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img,"Number Plate",(x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
             # 提取车牌区域并显示
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI", imgRoi)

    # 显示带有车牌检测结果的原始图像
    cv2.imshow("Result", img)
    # 如果按下 's' 键，保存检测到的车牌图像
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("img/NoPlate_"+str(count)+".jpg",imgRoi)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(150,265),cv2.FONT_HERSHEY_DUPLEX,
                    2,(0,0,255),2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count +=1







