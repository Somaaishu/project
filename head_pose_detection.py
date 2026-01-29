import cv2
import numpy as np

# Load face cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Head Pose Detection Started... Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    head_status = "No Face"

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_center_x = x + w // 2
        frame_center_x = frame.shape[1] // 2

        # Simple head direction logic
        if face_center_x < frame_center_x - 40:
            head_status = "Looking Left"
        elif face_center_x > frame_center_x + 40:
            head_status = "Looking Right"
        else:
            head_status = "Facing Forward"

        break  # consider only first face

    # Display status
    cv2.putText(frame, f"Head Pose: {head_status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Head Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
