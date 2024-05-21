from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('best.pt')

# Class names based on the model's configuration
classNames = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'Machinery', 'Vehicle'
]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width of the video
cap.set(4, 480)  # Set the height of the video

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = round(float(box.conf[0]), 2)  # Corrected rounding
            cls = int(box.cls[0])
            class_name = classNames[cls]

            cv2.putText(img, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
