import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
else:
    print("Camera is working")

while True:
    success, img = cap.read()
    if not success:
        print("Error: Cannot read frame")
        break
    
    cv2.imshow("Camera Feed", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
