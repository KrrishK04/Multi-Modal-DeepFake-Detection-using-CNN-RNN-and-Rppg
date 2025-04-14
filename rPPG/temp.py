import cv2

cap = cv2.VideoCapture(0)  # Change 0 to 1 if needed
if not cap.isOpened():
    print("Error: Could not access the camera.")
else:
    print("Camera accessed successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
