import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO model
model = YOLO('best.pt')

# Load class names from c.txt
with open("c.txt", "r") as file:
    class_list = file.read().split("\n")

count = 0
while True:
    im = picam2.capture_array()
    
    count += 1
    if count % 3 != 0:
        continue
    im = cv2.flip(im, -1)
    results = model.predict(im)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        
        # Validate class index against class list
        if d < len(class_list):
            c = class_list[d]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(im, f'{c}', (x1, y1), 1, 1)
        else:
            print(f"Detected class index {d} is out of range for class_list. Detected as {d}, which is not defined.")

    cv2.imshow("Camera", im)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
