import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp

from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

from mediapipe.tasks.python.vision.face_landmarker import _BaseOptions

# Download the face_landmarker_v2.task model if not present
MODEL_PATH = "face_landmarker_v2.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker_v2.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# Eye and iris landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# 3D model points for head pose estimation (nose tip, chin, left eye, right eye, left mouth, right mouth)
FACE_3D_IDX = [1, 152, 263, 33, 287, 57, 61, 291, 199]
FACE_2D_IDX = [1, 152, 263, 33, 287, 57, 61, 291, 199]


def eye_aspect_ratio(landmarks, eye_indices):
    points = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    vertical1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    vertical2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def iris_aspect_ratio(landmarks, iris_indices):
    # Use bounding box of iris for aspect ratio
    xs = [landmarks[i].x for i in iris_indices]
    ys = [landmarks[i].y for i in iris_indices]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return height / width if width > 0 else 0


# Head pose estimation using solvePnP
def head_pose(landmarks, w, h):
    # 3D model points (from MediaPipe face mesh)
    model_points = np.array([
        [0.0, 0.0, 0.0],      # Nose tip
        [0.0, -63.6, -12.5],  # Chin
        [-43.3, 32.7, -26.0], # Left eye left corner
        [43.3, 32.7, -26.0],  # Right eye right corner
        [-28.9, -28.9, -24.1],# Left mouth corner
        [28.9, -28.9, -24.1], # Right mouth corner
        [-61.6, -11.2, -39.5],# Left face edge
        [61.6, -11.2, -39.5], # Right face edge
        [0.0, -48.0, -50.0],  # Under nose
    ])
    image_points = np.array([
        [landmarks[i].x * w, landmarks[i].y * h] for i in FACE_2D_IDX
    ], dtype='double')
    # Camera internals
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype='double')
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0, 0, 0
    # Convert rotation vector to Euler angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    x = np.arctan2(rmat[2, 1], rmat[2, 2])
    y = np.arctan2(-rmat[2, 0], sy)
    z = np.arctan2(rmat[1, 0], rmat[0, 0])
    return np.degrees(x), np.degrees(y), np.degrees(z)



options = FaceLandmarkerOptions(
    base_options=_BaseOptions(model_asset_path=MODEL_PATH),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
face_landmarker = FaceLandmarker.create_from_options(options)


# --- Advanced Focus Detection ---
from collections import deque
import time

EAR_HISTORY = deque(maxlen=10)
IRIS_HISTORY = deque(maxlen=10)
POSE_HISTORY = deque(maxlen=10)
LOG_FILE = "focus_log.csv"

# Calibration
CALIBRATION_MODE = False
CALIBRATION_FRAMES = 100
calib_ear = []
calib_iris = []

def log_status(ts, ear, iris, pose, focus_status, confidence):
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{ear:.3f},{iris:.3f},{pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f},{focus_status},{confidence:.2f}\n")

print("Press 'c' to calibrate, 'q' to quit.")
cap = cv2.VideoCapture(0)
frame_count = 0
ear_thresh = 0.20
iris_thresh = 0.25
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    result = face_landmarker.detect(mp_image)
    h, w, _ = frame.shape
    ts = time.time()
    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2
        left_iris_ar = iris_aspect_ratio(landmarks, LEFT_IRIS)
        right_iris_ar = iris_aspect_ratio(landmarks, RIGHT_IRIS)
        iris_ar = (left_iris_ar + right_iris_ar) / 2
        x_angle, y_angle, z_angle = head_pose(landmarks, w, h)
        EAR_HISTORY.append(ear)
        IRIS_HISTORY.append(iris_ar)
        POSE_HISTORY.append((x_angle, y_angle, z_angle))
        # Calibration
        if CALIBRATION_MODE:
            calib_ear.append(ear)
            calib_iris.append(iris_ar)
            cv2.putText(frame, f"Calibrating... {len(calib_ear)}/{CALIBRATION_FRAMES}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if len(calib_ear) >= CALIBRATION_FRAMES:
                ear_thresh = np.mean(calib_ear) * 0.8
                iris_thresh = np.mean(calib_iris) * 0.8
                CALIBRATION_MODE = False
                calib_ear.clear()
                calib_iris.clear()
                print(f"Calibrated: EAR < {ear_thresh:.2f}, IRIS < {iris_thresh:.2f}")
            cv2.imshow("Focus Level Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        # Smoothing
        ear_smooth = np.mean(EAR_HISTORY)
        iris_smooth = np.mean(IRIS_HISTORY)
        pose_smooth = np.mean(POSE_HISTORY, axis=0)
        # Eye state
        if ear_smooth < ear_thresh and iris_smooth < iris_thresh:
            eye_state = "Closed"
        else:
            eye_state = "Open"
        # Head direction
        if abs(pose_smooth[1]) > 20:
            direction = "Looking Left" if pose_smooth[1] > 0 else "Looking Right"
        else:
            direction = "Looking Center"
        # Focus logic with confidence
        confidence = 1.0
        if eye_state == "Closed":
            focus_status = "Not Focused (Sleepy)"
            confidence -= 0.5
        elif direction != "Looking Center":
            focus_status = "Not Focused (Distracted)"
            confidence -= 0.3
        else:
            focus_status = "Focused"
        # Display
        cv2.putText(frame, f"EAR: {ear_smooth:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"IAR: {iris_smooth:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2)
        cv2.putText(frame, f"Pose Yaw: {pose_smooth[1]:.1f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame, f"Status: {focus_status}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        log_status(ts, ear_smooth, iris_smooth, pose_smooth, focus_status, confidence)
    else:
        cv2.putText(frame, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Focus Level Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        CALIBRATION_MODE = True
        calib_ear.clear()
        calib_iris.clear()
        print("Calibration started. Please look at the camera with eyes open.")
cap.release()
cv2.destroyAllWindows()
