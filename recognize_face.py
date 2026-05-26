import cv2
import mediapipe as mp
import json
import os
import numpy as np
from collections import deque

# ---------------- CONFIG ----------------
MODEL_PATH = "face_landmarker.task"
DATABASE_FOLDER = "database"

MATCH_THRESHOLD = 0.75
FRAME_BUFFER = 10

LEFT_EYE = 33
RIGHT_EYE = 263

# ---------------- MEDIAPIPE ----------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ---------------- NORMALIZATION ----------------
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]

    eye_distance = np.sqrt(
        (left_eye[0] - right_eye[0]) ** 2 +
        (left_eye[1] - right_eye[1]) ** 2
    )

    # avoid divide-by-zero
    if eye_distance == 0:
        eye_distance = 1

    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2

    normalized = []

    for x, y, z in landmarks:
        normalized.append([
            (x - center_x) / eye_distance,
            (y - center_y) / eye_distance,
            z / eye_distance
        ])

    return np.array(normalized)


# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


# ---------------- LOAD DATABASE ----------------
database = {}

for file in os.listdir(DATABASE_FOLDER):
    if file.endswith(".json"):

        path = os.path.join(
            DATABASE_FOLDER,
            file
        )

        with open(path, "r") as f:
            data = json.load(f)

            database[data["name"]] = np.array(
                data["landmarks"]
            )

print("Loaded:", list(database.keys()))

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

timestamp = 0
score_buffer = deque(maxlen=FRAME_BUFFER)

WEIGHTS = None

print("Press Q to quit")

while True:
    success, frame = cap.read()

    if not success:
        print("Camera failed")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 33

    person_name = "UNKNOWN"
    avg_score = 0

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        current_landmarks = []

        h, w, _ = frame.shape

        # ---------------- DRAW LANDMARKS ----------------
        for landmark in face_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

            current_landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # normalize
        current = normalize_landmarks(
            current_landmarks
        )

        # dynamic weights
        if WEIGHTS is None:

            num_points = len(current)

            WEIGHTS = np.ones(num_points)

            important_points = [
                33, 133, 263, 362,
                70, 63, 105, 66,
                336, 296, 334, 300,
                1, 6, 168, 197
            ]

            for idx in important_points:
                if idx < num_points:
                    WEIGHTS[idx] = 3.0

        best_name = None
        best_score = -1

        # ---------------- FACE MATCHING ----------------
        for name, stored in database.items():

            if len(stored) != len(current):
                continue

            weighted_current = (
                current *
                WEIGHTS[:, None]
            )

            weighted_stored = (
                stored *
                WEIGHTS[:, None]
            )

            score = cosine_similarity(
                weighted_current,
                weighted_stored
            )

            if score > best_score:
                best_score = score
                best_name = name

        # ---------------- FRAME AVERAGING ----------------
        if best_score != -1:
            score_buffer.append(best_score)

            avg_score = np.mean(
                score_buffer
            )

            if avg_score > MATCH_THRESHOLD:
                person_name = best_name

    # ---------------- UI ----------------
    cv2.putText(
        frame,
        f"Person: {person_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {avg_score:.3f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
