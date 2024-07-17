from ultralytics import YOLO
import cv2
import cvzone
import math

# url = 'http://192.168.29.16:81/stream'
cap = cv2.VideoCapture('ppe-1.mp4')

model = YOLO('ppe.pt')

myColor = (0,0,255)

# Load the exported NCNN model
# ncnn_model = YOLO("yolov8n_ncnn_model")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
while True:
    isTrue, frame = cap.read()
    results = model(frame, stream=True) 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)
            w, h = x2 -x1, y2 - y1
            # cvzone.cornerRect(frame, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.5:
               if currentClass == 'No-Hardhat' or currentClass == 'No-Mask' or currentClass == 'No-Safety Vest':
                 myColor = (0,0,255)
               elif currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                 myColor = (0,255,0)
               else:
                 myColor = (255,0,0)         
               cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale = 1, thickness=1, colorB=myColor, colorT=(255,255,255), colorR=myColor, offset=5)
               cv2.rectangle(frame, (x1,y1), (x2,y2), myColor, 3)

    cv2.imshow('Video', frame)
    cv2.waitKey(1)






