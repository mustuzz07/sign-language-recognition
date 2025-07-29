import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import numpy as np
import math
import time
import pickle
import pyttsx3
import threading

MODEL_PATH = "hand_model_cnn.keras"
LABELS_PATH = "hand_labels.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'rb') as f:
    labels = pickle.load(f)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# Initialize the pyttsx3 engine only once
engine = pyttsx3.init()

# Check if the run loop is already started
'''if not engine._inLoop:
    engine.startLoop()'''

offset = 20
img_size = 300
counter = 0
index = 0
message = []
weight_margen = 0

# Function for voice announcement
def announce_voice(engine, label):
    engine.say(label)
    engine.runAndWait()

while cap.isOpened():
    success, img = cap.read()
    img_output = img.copy()
    hands, img_output = detector.findHands(img)

    if hands:
        for hand in hands:
            x, y, w, h = hand["bbox"]

            imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

            imgCropped = img[y - offset: y + h + offset, x - offset: x + w + offset]

            # Check if imgCropped is empty
            if imgCropped.size == 0:
                continue

            imgCroppedShape = imgCropped.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = img_size / h
                new_w = math.ceil(k * w)
                img_resize = cv2.resize(imgCropped, (new_w, img_size))
                imgResizeShape = img_resize.shape
                wGap = math.ceil((img_size - new_w) / 2)
                imgWhite[:, wGap:new_w + wGap] = img_resize
                img_pred = imgWhite.copy()
                img_pred = cv2.resize(img_pred, (224, 224))
                img_pred = np.expand_dims(img_pred, axis=0)
                img_pred = np.vstack([img_pred])
                result = model.predict(img_pred)
                index = np.argmax(result)
            else:
                k = img_size / w
                new_h = math.ceil(k * h)
                img_resize = cv2.resize(imgCropped, (img_size, new_h))
                imgResizeShape = img_resize.shape
                hGap = math.ceil((img_size - new_h) / 2)
                imgWhite[hGap:new_h + hGap, :] = img_resize
                img_pred = imgWhite.copy()
                img_pred = cv2.resize(img_pred, (224, 224))
                img_pred = np.expand_dims(img_pred, axis=0)
                img_pred = np.vstack([img_pred])
                result = model.predict(img_pred)
                index = np.argmax(result)

            # Check if the index is within the range of labels
            if index < len(labels):
                label_text = labels[index]
                    
            else:
                label_text = "Unknown"

            cv2.imshow("Hand Cropped", imgCropped)
            cv2.imshow("White Image", imgWhite)

            cv2.rectangle(img_output, (x - offset - 8, y - offset - 50), (x - offset + 90, y - offset + 50),
                        (196, 196, 6), cv2.FILLED)
            cv2.putText(img_output, label_text, (x, y - 25), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

            cv2.rectangle(img_output, (x - offset, y - offset), (x + w + offset, y + h + offset), (196, 196, 6), 2)
            
            # Call voice announcement function in a separate thread
            threading.Thread(target=announce_voice, args=(engine, label_text,)).start()

    cv2.imshow("Image", img_output)
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
