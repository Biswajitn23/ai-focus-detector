import streamlit as st
import cv2
import numpy as np
import tempfile
import os
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
        st.error("Neither MediaPipe Tasks nor Solutions FaceMesh could be imported. Please install mediapipe.")
        st.stop()
from collections import deque
import time

# Download the face_landmarker_v2.task model if not present
MODEL_PATH = "face_landmarker_v2.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
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

st.title("Advanced Focus Level Detector")
st.write("Press 'Calibrate' to personalize thresholds. Webcam required.")

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

# Streamlit webcam input
frame_window = st.image([])
calibrate = st.button("Calibrate")
run = st.checkbox("Run Detector", value=True)


# Use longer smoothing window for more stability
EAR_HISTORY = deque(maxlen=20)
IRIS_HISTORY = deque(maxlen=20)
POSE_HISTORY = deque(maxlen=20)

CALIBRATION_MODE = False
CALIBRATION_FRAMES = 100
calib_ear = []
calib_iris = []
ear_thresh = 0.20
iris_thresh = 0.25

if 'ear_thresh' not in st.session_state:
    st.session_state['ear_thresh'] = 0.20
if 'iris_thresh' not in st.session_state:
    st.session_state['iris_thresh'] = 0.25

if calibrate:
    CALIBRATION_MODE = True
    calib_ear.clear()
    calib_iris.clear()
    st.info("Calibration started. Please look at the camera with eyes open.")

cap = cv2.VideoCapture(0)
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("No webcam frame.")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect with Tasks API if available, otherwise use Solutions FaceMesh
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
    h, w, _ = frame.shape
    # Ambient light check (simple mean pixel value)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 40:
        cv2.putText(frame, "Low light: accuracy reduced", (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if landmarks is not None:
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        left_iris_ar = iris_aspect_ratio(landmarks, LEFT_IRIS)
        right_iris_ar = iris_aspect_ratio(landmarks, RIGHT_IRIS)
        # Require both eyes closed for "sleepy"
        both_eyes_closed = (left_ear < st.session_state['ear_thresh'] and left_iris_ar < st.session_state['iris_thresh'] and
                            right_ear < st.session_state['ear_thresh'] and right_iris_ar < st.session_state['iris_thresh'])
        ear = (left_ear + right_ear) / 2
        iris_ar = (left_iris_ar + right_iris_ar) / 2
        x_angle, y_angle, z_angle = head_pose(landmarks, w, h)
        EAR_HISTORY.append(ear)
        IRIS_HISTORY.append(iris_ar)
        POSE_HISTORY.append((x_angle, y_angle, z_angle))
        if CALIBRATION_MODE:
            calib_ear.append(ear)
            calib_iris.append(iris_ar)
            st.info(f"Calibrating... {len(calib_ear)}/{CALIBRATION_FRAMES}")
            if len(calib_ear) >= CALIBRATION_FRAMES:
                st.session_state['ear_thresh'] = np.mean(calib_ear) * 0.8
                st.session_state['iris_thresh'] = np.mean(calib_iris) * 0.8
                CALIBRATION_MODE = False
                calib_ear.clear()
                calib_iris.clear()
                st.success(f"Calibrated: EAR < {st.session_state['ear_thresh']:.2f}, IRIS < {st.session_state['iris_thresh']:.2f}")
            display_frame = cv2.flip(frame, 1)
            frame_window.image(display_frame, channels="BGR")
            continue
        ear_smooth = np.mean(EAR_HISTORY)
        iris_smooth = np.mean(IRIS_HISTORY)
        pose_smooth = np.mean(POSE_HISTORY, axis=0)
        # (face_confidence not available in FaceLandmarker API)
        # Eye state
        if both_eyes_closed:
            eye_state = "Closed"
        else:
            eye_state = "Open"
        # Head direction (ignore if face turned too far)
        if abs(pose_smooth[1]) > 30:
            direction = "Face turned away"
        elif abs(pose_smooth[1]) > 20:
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
    else:
        cv2.putText(frame, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    display_frame = cv2.flip(frame, 1)
    frame_window.image(display_frame, channels="BGR")
    if not run:
        break
cap.release()
st.write("App stopped. Uncheck 'Run Detector' to stop webcam.")
