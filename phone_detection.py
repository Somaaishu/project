import cv2

# Load pre-trained Haar Cascade (not perfect but works)
phone_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    phones = phone_cascade.detectMultiScale(gray, 1.3, 5)

    if len(phones) > 0:
        cv2.putText(frame, "ALERT: Possible Phone Detected!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Phone Detection (Basic)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


