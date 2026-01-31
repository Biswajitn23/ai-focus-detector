import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import csv

# Try to use MediaPipe Tasks API (FaceLandmarker). If unavailable, fall back to
# MediaPipe Solutions FaceMesh which is more widely available in wheels.
USE_TASKS = False
FaceLandmarker = None
FaceLandmarkerOptions = None
_BaseOptions = None
Image = None
ImageFormat = None
face_detector = None

try:
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python.vision.face_landmarker import _BaseOptions
    try:
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
    except Exception:
        from mediapipe.tasks.python.vision import Image, ImageFormat
    USE_TASKS = True
except Exception:
    # fallback to solutions FaceMesh
    try:
        from mediapipe import solutions as mps
        USE_TASKS = False
        mps_face_mesh = mps.face_mesh
    except Exception:
        raise ImportError("Neither MediaPipe Tasks nor Solutions FaceMesh could be imported. Please install mediapipe.")
import os

MODEL_PATH = "face_landmarker_v2.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task"

# Download model if not
def download_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

if USE_TASKS:
    download_model()

# Initialize detector depending on available API
if USE_TASKS:
    options_init = FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    face_landmarker = FaceLandmarker.create_from_options(options_init)
else:
    # use Solutions FaceMesh as fallback
    face_mesh = mps_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FACE_3D_IDX = [1, 152, 263, 33, 287, 57, 61, 291, 199]
FACE_2D_IDX = [1, 152, 263, 33, 287, 57, 61, 291, 199]

# smoothing / calibration / debounce state (module-level persists across calls)
EAR_HISTORY = deque(maxlen=10)
IRIS_HISTORY = deque(maxlen=10)
POSE_HISTORY = deque(maxlen=10)

CALIBRATION_MODE = False
CALIBRATION_FRAMES = 100
calib_ear = []
calib_iris = []
ear_thresh = 0.20
iris_thresh = 0.25

CONSEC_FRAMES_REQUIRED = 5
closed_count = 0
open_count = 0

LOG_FILE = "focus_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts","ear","iris","yaw","pitch","roll","status","confidence"])

def eye_aspect_ratio(landmarks, eye_indices):
    points = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    vertical1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    vertical2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def iris_aspect_ratio(landmarks, iris_indices):
    xs = [landmarks[i].x for i in iris_indices]
    ys = [landmarks[i].y for i in iris_indices]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return height / width if width > 0 else 0


def head_pose(landmarks, w, h):
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
        [-61.6, -11.2, -39.5],
        [61.6, -11.2, -39.5],
        [0.0, -48.0, -50.0],
    ])
    image_points = np.array([
        [landmarks[i].x * w, landmarks[i].y * h] for i in FACE_2D_IDX
    ], dtype='double')
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
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    x = np.arctan2(rmat[2, 1], rmat[2, 2])
    y = np.arctan2(-rmat[2, 0], sy)
    z = np.arctan2(rmat[1, 0], rmat[0, 0])
    return np.degrees(x), np.degrees(y), np.degrees(z)

def detect_focus(frame):
    # Normalize input from Gradio: accept None, file path, PIL image, or numpy array.
    from PIL import Image as PILImage

    def to_rgb(img):
        if img is None:
            return None
        if isinstance(img, str):
            arr = cv2.imread(img)
            if arr is None:
                return None
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if isinstance(img, np.ndarray):
            if img.size == 0:
                return None
            # Gradio typically supplies RGB numpy arrays for images/webcam.
            return img
        if isinstance(img, PILImage.Image):
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGB)
        return None

    rgb_frame = to_rgb(frame)
    if rgb_frame is None:
        # return a small blank image with an error message
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "No frame", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return blank
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

    if landmarks is not None:
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        left_iris_ar = iris_aspect_ratio(landmarks, LEFT_IRIS)
        right_iris_ar = iris_aspect_ratio(landmarks, RIGHT_IRIS)
        ear = (left_ear + right_ear) / 2
        iris_ar = (left_iris_ar + right_iris_ar) / 2
        h, w, _ = rgb_frame.shape
        yaw, pitch, roll = head_pose(landmarks, w, h)
        # smoothing
        EAR_HISTORY.append(ear)
        IRIS_HISTORY.append(iris_ar)
        POSE_HISTORY.append((yaw, pitch, roll))
        ear_smooth = np.mean(EAR_HISTORY)
        iris_smooth = np.mean(IRIS_HISTORY)
        pose_smooth = np.mean(POSE_HISTORY, axis=0)

        # Auto-calibration during first frames if not manually calibrated
        global CALIBRATION_MODE, calib_ear, calib_iris, ear_thresh, iris_thresh, closed_count, open_count
        if CALIBRATION_MODE or (len(calib_ear) < CALIBRATION_FRAMES and ear_smooth > 0.25):
            CALIBRATION_MODE = True
            calib_ear.append(ear_smooth)
            calib_iris.append(iris_smooth)
            if len(calib_ear) >= CALIBRATION_FRAMES:
                ear_thresh = np.mean(calib_ear) * 0.8
                iris_thresh = np.mean(calib_iris) * 0.8
                CALIBRATION_MODE = False
                calib_ear.clear()
                calib_iris.clear()

        # Determine eye state with debounce
        both_eyes_closed = (ear_smooth < ear_thresh and iris_smooth < iris_thresh)
        # head pose gating
        turned_away = abs(pose_smooth[1]) > 30
        if both_eyes_closed:
            closed_count += 1
            open_count = 0
        else:
            open_count += 1
            closed_count = 0

        is_sleepy = closed_count >= CONSEC_FRAMES_REQUIRED
        # Compose status
        if is_sleepy:
            status = "Not Focused (Sleepy)"
            confidence = 0.5
        elif turned_away:
            status = "Not Focused (Turned Away)"
            confidence = 0.6
        else:
            status = "Focused"
            confidence = 1.0

        # logging
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), ear_smooth, iris_smooth, pose_smooth[0], pose_smooth[1], pose_smooth[2], status, confidence])
        except Exception:
            pass
        # Draw status
        if ear < 0.20 and iris_ar < 0.25:
            status = "Sleepy/Not Focused"
            color = (0, 0, 255)
        else:
            status = "Focused"
            color = (0, 255, 0)
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"Status: {status}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return bgr
    else:
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return bgr

# Create a Gradio Image input component compatible with multiple gradio versions
try:
    img_input = gr.Image(source="webcam", tool=None)
except TypeError:
    try:
        img_input = gr.inputs.Image(source="webcam")
    except Exception:
        img_input = gr.Image()

demo = gr.Interface(
    fn=detect_focus,
    inputs=img_input,
    outputs=gr.Image(),
    live=True,
    title="Focus Detector (Webcam)",
    description="Detects if you are focused or sleepy using your webcam."
)

demo.launch(share=True, server_name="0.0.0.0")
