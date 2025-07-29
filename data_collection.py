import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

detector = HandDetector(maxHands=2)

# Additional values for ROI
offset = 50
img_size = 300
counter = 0
textdata = input("Enter sign text:")

def create_folder(folder_path):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

# Example usage:
folder_path = "data/" + textdata
create_folder(folder_path)

FOLDER = folder_path
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Error: Cannot read frame")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        #print(f"x: {x}, y: {y}, w: {w}, h: {h}")  # Debug print statement
        
        # Crop the hand region
        imgCropped = img[y - offset: y + h + offset, x - offset: x + w + offset]

        # Ensure imgCropped is not empty
        if imgCropped.size == 0:
            print("Warning: Cropped image is empty")
            continue

        imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255
        imgCroppedShape = imgCropped.shape
        aspectRatio = h / w

        # Padding white for resized image
        if aspectRatio > 1:
            k = img_size / h
            new_w = math.ceil(k * w)
            img_resize = cv2.resize(imgCropped, (new_w, img_size))
            wGap = math.ceil((img_size - new_w) / 2)
            imgWhite[:, wGap:new_w + wGap] = img_resize
        else:
            k = img_size / w
            new_h = math.ceil(k * h)
            img_resize = cv2.resize(imgCropped, (img_size, new_h))
            hGap = math.ceil((img_size - new_h) / 2)
            imgWhite[hGap:new_h + hGap, :] = img_resize

        cv2.imshow("Hand Cropped", imgCropped)
        cv2.imshow("White Image", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{FOLDER}/image_{time.time()}.jpg", imgWhite)
        print(f"Image saved. Total images: {counter}")

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
