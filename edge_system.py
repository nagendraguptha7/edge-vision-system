import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (lightweight, fast)
model = YOLO("yolov8n.pt")

# Use webcam or video file
cap = cv2.VideoCapture(0)   # change to video path if needed

object_ids = set()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run object detection
    results = model(frame, conf=0.4)[0]

    detections = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections += 1

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f'{conf:.2f}', (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Simple analytics
    if detections > 8:
        cv2.putText(frame, "CONGESTION ALERT", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.putText(frame, f'Objects: {detections}', (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Edge AI System Prototype", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
