import cv2
import mediapipe as mp
import numpy as np
from pygame import mixer

# Initialize mixer and alarm sound
mixer.init()
ALARM_SOUND = mixer.Sound("alarm.wav")

# EAR threshold and frame tracking
EYE_AR_THRESHOLD = 0.25
DROWSY_FRAMES_REQUIRED = 10

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Indices for left and right eye landmarks (from MediaPipe Face Mesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def euclidean_dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(eye_landmarks):
    # Eye landmarks: [left_corner, top_left, top_right, right_corner, bottom_right, bottom_left]
    A = euclidean_dist(eye_landmarks[1], eye_landmarks[5])  # vertical
    B = euclidean_dist(eye_landmarks[2], eye_landmarks[4])  # vertical
    C = euclidean_dist(eye_landmarks[0], eye_landmarks[3])  # horizontal
    return (A + B) / (2.0 * C)

# Start webcam
cap = cv2.VideoCapture(0)
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            def get_landmark_coords(indices):
                return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

            left_eye = get_landmark_coords(LEFT_EYE_INDICES)
            right_eye = get_landmark_coords(RIGHT_EYE_INDICES)

            for x, y in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EYE_AR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= DROWSY_FRAMES_REQUIRED:
                    try:
                        ALARM_SOUND.play()
                    except:
                        pass
                    cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                frame_counter = 0
                try:
                    ALARM_SOUND.stop()
                except:
                    pass

    cv2.imshow("Drowsiness Detection (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
