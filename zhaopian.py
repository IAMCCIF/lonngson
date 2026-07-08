import cv2 as cv
import os
import time

# 模型路径
model_bin = "MobileNetSSD_deploy.caffemodel"
config_text = "MobileNetSSD_deploy.prototxt.txt"

if not os.path.exists(model_bin):
    print(f"错误: 找不到模型 {model_bin}")
    exit(1)
if not os.path.exists(config_text):
    print(f"错误: 找不到配置 {config_text}")
    exit(1)

# 类别
objName = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# 加载模型
net = cv.dnn.readNetFromCaffe(config_text, model_bin)

# 屏蔽GStreamer报错
os.environ["GST_DEBUG"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY"] = "V4L"

# 打开摄像头
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("摄像头打开失败！")
    exit(1)

print("摄像头启动成功，按 Q 退出")

# 帧率计算
fps = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # 计算FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # 模型推理
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # 绘制检测框
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{objName[class_id]} {confidence:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示FPS
    cv.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv.imshow("Loongson Object Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
