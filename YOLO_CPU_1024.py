import cv2
import numpy as np
import sys
import os

def resource_path(relative_path):
    """ リソースへの絶対パスを取得する。 """
    try:
        # PyInstaller はテンポラリフォルダを作成し、パスを _MEIPASS に格納します。
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# YOLOv4-tinyの設定ファイルと重みファイルのパス
weightsPath = resource_path('yolov4-tiny.weights')
configPath = resource_path('yolov4-tiny.cfg')

net = cv2.dnn.readNet(weightsPath, configPath)

# cocoデータセットのクラス名を読み込む
cocoPath = resource_path('coco.names')

with open(cocoPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# カメラからのリアルタイムビデオストリームを開始
cap = cv2.VideoCapture(0)
# Width: 640.0, Height: 480.0

while True:
    _, frame = cap.read()

    # フレームを左右反転する
    frame = cv2.flip(frame, 1)

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 出力画像を1024x960にリサイズする。
    frame = cv2.resize(frame, (1280, 960))

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
