import cv2
import csv
import time
from datetime import datetime

# ------------------------------
# OpenCV Haar Cascades
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ------------------------------
# Webcam
# ------------------------------
cap = cv2.VideoCapture(0)

# ------------------------------
# Timer variables
# ------------------------------
looking_away_start = None
total_looking_away_time = 0

# ------------------------------
# CSV Log File
# ------------------------------
log_file = "proctoring_log_with_time.csv"
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Face Status", "Eye Status", "Away Duration (sec)"])

print("AI Proctoring Started... Press ESC to stop")

# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_status = "No Face Detected"
    eye_status = "No Face Detected"
    alert_text = ""

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ------------------------------
    # Face Detection
    # ------------------------------
    if len(faces) == 0:
        face_status = "No Face Detected"
        alert_text = "ALERT: No Face!"
    elif len(faces) > 1:
        face_status = "Multiple Faces Detected"
        alert_text = "ALERT: Multiple Faces!"
    else:
        face_status = "Normal"
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # ------------------------------
        # Eye Tracking
        # ------------------------------
        if len(eyes) >= 2:
            eye_status = "Normal"
        else:
            eye_status = "Looking Away"
            alert_text = "ALERT: Looking Away!"

    # ------------------------------
    # Time Tracking
    # ------------------------------
    if eye_status == "Looking Away":
        if looking_away_start is None:
            looking_away_start = time.time()
    else:
        if looking_away_start is not None:
            total_looking_away_time += time.time() - looking_away_start
            looking_away_start = None

    # ------------------------------
    # Log Suspicious Events
    # ------------------------------
    if face_status != "Normal" or eye_status != "Normal":
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        away_time = round(total_looking_away_time, 2)
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, face_status, eye_status, away_time])

    # ------------------------------
    # Display on Screen
    # ------------------------------
    cv2.putText(frame, f"Face Status: {face_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Eye Status: {eye_status}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"Looking Away Time: {int(total_looking_away_time)} sec",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

    if alert_text:
        cv2.putText(frame, alert_text, (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
cv2.destroyAllWindows()

print("Session Ended")
print(f"Total Looking Away Time: {int(total_looking_away_time)} seconds")
print(f"Logs saved in: {log_file}")

