import cv2
import mediapipe as mp
import json
import os
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "face_landmarker.task"
DATABASE_FOLDER = "database"

NUM_SAMPLES = 50

os.makedirs(DATABASE_FOLDER, exist_ok=True)

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
LEFT_EYE = 33
RIGHT_EYE = 263


def normalize_landmarks(landmarks):
    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]

    eye_distance = np.sqrt(
        (left_eye[0] - right_eye[0]) ** 2 +
        (left_eye[1] - right_eye[1]) ** 2
    )

    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2

    normalized = []

    for x, y, z in landmarks:
        normalized.append([
            (x - center_x) / eye_distance,
            (y - center_y) / eye_distance,
            z / eye_distance
        ])

    return normalized


# ---------------- INPUT ----------------
person_name = input("Enter person name: ").strip().lower()

cap = cv2.VideoCapture(0)

timestamp = 0
samples = []

print("Move face slightly while recording")
print("Press S to start capture")

started = False

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 33

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]

        current_landmarks = []

        h, w, _ = frame.shape

        for landmark in face_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            current_landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        if started and len(samples) < NUM_SAMPLES:
            normalized = normalize_landmarks(current_landmarks)
            samples.append(normalized)

    cv2.putText(
        frame,
        f"Samples: {len(samples)}/{NUM_SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        started = True

    elif key == ord("q"):
        break

    if len(samples) >= NUM_SAMPLES:
        break

cap.release()
cv2.destroyAllWindows()

# Average landmarks
avg_landmarks = np.mean(samples, axis=0).tolist()

save_path = os.path.join(
    DATABASE_FOLDER,
    f"{person_name}.json"
)

with open(save_path, "w") as f:
    json.dump(
        {
            "name": person_name,
            "landmarks": avg_landmarks
        },
        f,
        indent=4
    )

print(f"Saved {person_name}")
