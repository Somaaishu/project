import cv2
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

cheat_start_time = None
cheat_duration = 0

print("AI Proctoring System Started... Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "Normal"

    # 🔴 MULTIPLE FACE LOGIC
    if len(faces) == 0:
        status = "No Face Detected"

    elif len(faces) > 1:
        status = "Multiple Faces Detected"

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    else:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_center_x = x + w // 2
        frame_center_x = frame.shape[1] // 2

        if face_center_x < frame_center_x - 40:
            status = "Looking Left"
        elif face_center_x > frame_center_x + 40:
            status = "Looking Right"
        else:
            status = "Facing Forward"

    # ⏱ CHEATING TIMER
    if status != "Facing Forward":
        if cheat_start_time is None:
            cheat_start_time = time.time()
        cheat_duration = int(time.time() - cheat_start_time)
    else:
        cheat_start_time = None
        cheat_duration = 0

    # 🚨 ALERT
    if cheat_duration > 5:
        cv2.putText(frame, "ALERT: Suspicious Behavior!",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

    # 📺 DISPLAY
    cv2.putText(frame, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cheat Time: {cheat_duration}s", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

