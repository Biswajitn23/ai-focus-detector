import argparse
import cv2
import numpy as np
import os
import time
import urllib.request
import mediapipe as mp

# Prefer MediaPipe Tasks FaceLandmarker; fall back to Solutions FaceMesh
USE_TASKS = False
FaceLandmarker = None
FaceLandmarkerOptions = None
_BaseOptions = None
Image = None
ImageFormat = None
try:
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.face_landmarker import _BaseOptions
    try:
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
    except Exception:
        from mediapipe.tasks.python.vision import Image, ImageFormat
    USE_TASKS = True
except Exception:
    try:
        from mediapipe import solutions as mps
        mps_face_mesh = mps.face_mesh
        USE_TASKS = False
    except Exception:
        raise ImportError("Neither MediaPipe Tasks nor Solutions FaceMesh could be imported. Please install mediapipe.")

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
    # protect against degenerate horizontal distance
    if horizontal <= 1e-6:
        return 0.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def iris_aspect_ratio(landmarks, iris_indices):
    # Use bounding box of iris for aspect ratio
    xs = [landmarks[i].x for i in iris_indices]
    ys = [landmarks[i].y for i in iris_indices]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return height / width if width > 1e-6 else 0


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
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        return 0.0, 0.0, 0.0
    if not success:
        return 0.0, 0.0, 0.0
    # Convert rotation vector to Euler angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    x = np.arctan2(rmat[2, 1], rmat[2, 2])
    y = np.arctan2(-rmat[2, 0], sy)
    z = np.arctan2(rmat[1, 0], rmat[0, 0])
    return np.degrees(x), np.degrees(y), np.degrees(z)



if USE_TASKS:
    options = FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)
else:
    face_mesh = mps_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)


# --- Advanced Focus Detection (improvements) ---
from collections import deque
import time

# Tunable params
HISTORY_LEN = 30
EMA_ALPHA = 0.35
MIN_FACE_SCALE = 0.08

# Default yaw thresholds (degrees)
YAW_FACE_AWAY = 45
YAW_DISTRACT = 20

# Parse command-line args
parser = argparse.ArgumentParser(description="Focus detector options")
parser.add_argument("--min-face-scale", type=float, default=MIN_FACE_SCALE, help="Minimum normalized face width/height to accept")
parser.add_argument("--yaw-away", type=float, default=YAW_FACE_AWAY, help="Yaw deg considered 'face turned away'")
parser.add_argument("--yaw-distract", type=float, default=YAW_DISTRACT, help="Yaw deg considered 'distracted'")
parser.add_argument("--calib-frames", type=int, default=120, help="Calibration frames")
parser.add_argument("--test-seconds", type=int, default=0, help="Run for this many seconds then exit (0 = run until quit)")
parser.add_argument("--verbose", action="store_true", help="Verbose console logging each frame")
args = parser.parse_args()

# apply args
MIN_FACE_SCALE = args.min_face_scale
YAW_FACE_AWAY = args.yaw_away
YAW_DISTRACT = args.yaw_distract
CALIBRATION_FRAMES = args.calib_frames

EAR_HISTORY = deque(maxlen=HISTORY_LEN)
IRIS_HISTORY = deque(maxlen=HISTORY_LEN)
POSE_HISTORY = deque(maxlen=HISTORY_LEN)
LOG_FILE = "focus_log.csv"

# Calibration
CALIBRATION_MODE = False
calib_ear = []
calib_iris = []

# EMA state
ear_ema = None
iris_ema = None
pose_ema = None

def log_status(ts, ear, iris, pose, focus_status, confidence):
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{ear:.3f},{iris:.3f},{pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f},{focus_status},{confidence:.2f}\n")

def face_scale_from_landmarks(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width, height

print("Press 'c' to calibrate, 'q' to quit.")
cap = cv2.VideoCapture(0)
start_time = time.time()
frame_count = 0
ear_thresh = 0.20
iris_thresh = 0.25
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    # detect via Tasks API or Solutions FaceMesh
    landmarks = None
    if USE_TASKS:
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        result = face_landmarker.detect(mp_image)
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
    else:
        results = face_mesh.process(rgb_frame)
        if results and results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
    ts = time.time()
    if landmarks is not None:
        # ignore tiny faces (too far / unreliable)
        face_w_norm, face_h_norm = face_scale_from_landmarks(landmarks)
        if face_w_norm < MIN_FACE_SCALE and face_h_norm < MIN_FACE_SCALE:
            cv2.putText(frame, "Face too small", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            disp = cv2.flip(frame, 1)
            cv2.imshow("Focus Level Detector", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2
        left_iris_ar = iris_aspect_ratio(landmarks, LEFT_IRIS)
        right_iris_ar = iris_aspect_ratio(landmarks, RIGHT_IRIS)
        iris_ar = (left_iris_ar + right_iris_ar) / 2
        x_angle, y_angle, z_angle = head_pose(landmarks, w, h)

        # EMA smoothing (more robust than simple mean)
        if ear_ema is None:
            ear_ema = ear
            iris_ema = iris_ar
            pose_ema = np.array([x_angle, y_angle, z_angle])
        else:
            ear_ema = EMA_ALPHA * ear + (1 - EMA_ALPHA) * ear_ema
            iris_ema = EMA_ALPHA * iris_ar + (1 - EMA_ALPHA) * iris_ema
            pose_ema = EMA_ALPHA * np.array([x_angle, y_angle, z_angle]) + (1 - EMA_ALPHA) * pose_ema

        EAR_HISTORY.append(ear)
        IRIS_HISTORY.append(iris_ar)
        POSE_HISTORY.append((x_angle, y_angle, z_angle))

        # Calibration: use low-percentile threshold for robustness
        if CALIBRATION_MODE:
            calib_ear.append(ear)
            calib_iris.append(iris_ar)
            if len(calib_ear) >= CALIBRATION_FRAMES:
                ear_thresh = max(0.04, np.percentile(calib_ear, 10) * 0.95)
                iris_thresh = max(0.05, np.percentile(calib_iris, 10) * 0.95)
                CALIBRATION_MODE = False
                calib_ear.clear()
                calib_iris.clear()
                print(f"Calibrated: EAR < {ear_thresh:.2f}, IRIS < {iris_thresh:.2f}")
            disp = cv2.flip(frame, 1)
            cv2.putText(disp, f"Calibrating... {len(calib_ear)}/{CALIBRATION_FRAMES}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.imshow("Focus Level Detector", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        ear_smooth = ear_ema
        iris_smooth = iris_ema
        pose_smooth = pose_ema if pose_ema is not None else np.array([0.0, 0.0, 0.0])

        # Eye state
        if ear_smooth < ear_thresh and iris_smooth < iris_thresh:
            eye_state = "Closed"
        else:
            eye_state = "Open"

        # Head direction (ignore if face turned too far)
        if abs(pose_smooth[1]) > YAW_FACE_AWAY:
            direction = "Face turned away"
        elif abs(pose_smooth[1]) > YAW_DISTRACT:
            direction = "Looking Left" if pose_smooth[1] > 0 else "Looking Right"
        else:
            direction = "Looking Center"

        # Focus logic with confidence
        confidence = 1.0
        if eye_state == "Closed":
            focus_status = "Not Focused (Sleepy)"
            confidence -= 0.5
        elif direction == "Face turned away":
            focus_status = "Not Focused (Not visible)"
            confidence -= 0.5
        elif direction != "Looking Center":
            focus_status = "Not Focused (Distracted)"
            confidence -= 0.3
        else:
            focus_status = "Focused"

        # Prepare display text and then render on flipped image so text is not mirrored
        disp = cv2.flip(frame, 1)
        cv2.putText(disp, f"EAR: {ear_smooth:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(disp, f"IAR: {iris_smooth:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2)
        cv2.putText(disp, f"Pose Yaw: {pose_smooth[1]:.1f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(disp, f"Status: {focus_status}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(disp, f"Confidence: {confidence:.2f}", (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Debug overlay: yaw, ear, iris, normalized face width and yaw-away threshold
        cv2.putText(disp, f"Yaw:{pose_smooth[1]:.1f} EAR:{ear_smooth:.2f} IAR:{iris_smooth:.2f} faceW:{face_w_norm:.2f} yawAway:{YAW_FACE_AWAY}", (30,230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2)
        if args.verbose:
            print(f"ts={ts:.3f} yaw={pose_smooth[1]:.2f} ear={ear_smooth:.3f} iar={iris_smooth:.3f} faceW={face_w_norm:.3f} status={focus_status} conf={confidence:.2f}")
        log_status(ts, ear_smooth, iris_smooth, pose_smooth, focus_status, confidence)
    else:
        cv2.putText(frame, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # If we didn't already create a display image (non-landmarks branch), create and draw non-face text
    if 'disp' not in locals():
        disp = cv2.flip(frame, 1)
        if landmarks is None:
            cv2.putText(disp, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Focus Level Detector", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        CALIBRATION_MODE = True
        calib_ear.clear()
        calib_iris.clear()
        ear_ema = None
        iris_ema = None
        pose_ema = None
        print("Calibration started. Please look at the camera with eyes open.")
    # timed test mode: exit after configured seconds
    if args.test_seconds and (time.time() - start_time) > args.test_seconds:
        print(f"Test time {args.test_seconds}s reached, exiting")
        break
cap.release()
cv2.destroyAllWindows()
