import cv2
import mediapipe as mp

# ------------------------------
# Setup MediaPipe Face Detection
# ------------------------------
# Use fallback to OpenCV if MediaPipe .solutions not available
try:
    mp_face_detection = mp.solutions.face_detection.FaceDetection
    use_mediapipe = True
except AttributeError:
    use_mediapipe = False

# ------------------------------
# OpenCV Haar Cascades
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ------------------------------
# MediaPipe Face Mesh for Eyes (optional)
# ------------------------------
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    use_face_mesh = True
except AttributeError:
    use_face_mesh = False

# ------------------------------
# Webcam
# ------------------------------
cap = cv2.VideoCapture(0)

if use_mediapipe:
    face_detector = mp_face_detection(model_selection=0, min_detection_confidence=0.5)

if use_face_mesh:
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    status_face = "No Face Detected"
    status_eye = "No Face Detected"

    # ------------------------------
    # Face Detection
    # ------------------------------
    if use_mediapipe:
        results = face_detector.process(frame_rgb)
        if results.detections:
            if len(results.detections) > 1:
                status_face = "Multiple Faces Detected"
            else:
                status_face = "Normal"
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            status_face = "No Face Detected"
        elif len(faces) == 1:
            status_face = "Normal"
        else:
            status_face = "Multiple Faces Detected"

    # ------------------------------
    # Eye Tracking
    # ------------------------------
    if use_face_mesh:
        mesh_results = face_mesh.process(frame_rgb)
        if mesh_results.multi_face_landmarks:
            status_eye = "Normal"
            for face_landmarks in mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)
                left_eye = [face_landmarks.landmark[i] for i in [33, 133]]
                right_eye = [face_landmarks.landmark[i] for i in [362, 263]]
                left_ratio = abs(left_eye[0].x - left_eye[1].x)
                right_ratio = abs(right_eye[0].x - right_eye[1].x)
                if left_ratio < 0.03 or right_ratio < 0.03:
                    status_eye = "Looking Away"
    else:
        # Fallback using OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                status_eye = "Normal"
            else:
                status_eye = "Looking Away"

    # ------------------------------
    # Display Status
    # ------------------------------
    cv2.putText(frame, f"Face Status: {status_face}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Eye Status: {status_eye}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
