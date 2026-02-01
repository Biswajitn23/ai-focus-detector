import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import csv
import os
import joblib

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
import joblib

# Load trained model if available
MODEL_FILE = "focus_model.pkl"
trained_clf = None
if os.path.exists(MODEL_FILE):
    try:
        trained_clf = joblib.load(MODEL_FILE)
        print(f"Loaded trained model from {MODEL_FILE}")
    except Exception as e:
        print(f"Failed to load model: {e}")

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
STATIC_EAR_SLEEPY = 0.18
STATIC_IRIS_SLEEPY = 0.22

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

def detect_focus(frame, detailed=False, return_metrics=False, use_trained=True):
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
        history_len = len(EAR_HISTORY)
        both_eyes_closed = (ear_smooth < ear_thresh and iris_smooth < iris_thresh)
        # head pose gating
        turned_away = abs(pose_smooth[1]) > 30
        if both_eyes_closed:
            closed_count += 1
            open_count = 0
        else:
            open_count += 1
            closed_count = 0

        # For single/very few frames (e.g., uploaded images), use fixed one-shot thresholds
        if history_len <= 2:
            is_sleepy = (ear_smooth < STATIC_EAR_SLEEPY and iris_smooth < STATIC_IRIS_SLEEPY)
        else:
            is_sleepy = closed_count >= CONSEC_FRAMES_REQUIRED
        
        # Compute gaze direction (iris offset within eye bounds)
        left_iris_x = np.mean([landmarks[i].x for i in LEFT_IRIS])
        right_iris_x = np.mean([landmarks[i].x for i in RIGHT_IRIS])
        left_eye_x_center = np.mean([landmarks[i].x for i in LEFT_EYE])
        right_eye_x_center = np.mean([landmarks[i].x for i in RIGHT_EYE])
        left_eye_w = max([landmarks[i].x for i in LEFT_EYE]) - min([landmarks[i].x for i in LEFT_EYE])
        right_eye_w = max([landmarks[i].x for i in RIGHT_EYE]) - min([landmarks[i].x for i in RIGHT_EYE])
        if left_eye_w < 1e-6: left_eye_w = 1e-6
        if right_eye_w < 1e-6: right_eye_w = 1e-6
        left_offset = (left_iris_x - left_eye_x_center) / left_eye_w
        right_offset = (right_iris_x - right_eye_x_center) / right_eye_w
        gaze_x = (left_offset + right_offset) / 2.0
        
        # Use trained model if available
        if use_trained and trained_clf is not None:
            try:
                features = np.array([ear_smooth, iris_smooth, pose_smooth[0], pose_smooth[1], pose_smooth[2], gaze_x])
                pred = trained_clf.predict([features])[0]
                pred_proba = trained_clf.predict_proba([features])[0]
                if pred == 1:
                    status = "Focused"
                    confidence = float(pred_proba[1])
                else:
                    status = "Not Focused"
                    confidence = float(pred_proba[0])
            except Exception as e:
                print(f"Model prediction error: {e}")
                # Fallback to heuristic if model fails
                if is_sleepy:
                    status = "Not Focused (Sleepy)"
                    confidence = 0.5
                elif turned_away:
                    status = "Not Focused (Turned Away)"
                    confidence = 0.6
                else:
                    status = "Focused"
                    confidence = 1.0
        else:
            # Fallback to heuristic
            if is_sleepy:
                status = "Not Focused (Sleepy)"
                confidence = 0.5
            elif turned_away:
                status = "Not Focused (Turned Away)"
                confidence = 0.6
            else:
                status = "Focused"
                confidence = 1.0

        metrics = dict(
            ear=float(ear_smooth),
            iris=float(iris_smooth),
            yaw=float(pose_smooth[0]),
            pitch=float(pose_smooth[1]),
            roll=float(pose_smooth[2]),
            ear_thresh=float(ear_thresh),
            iris_thresh=float(iris_thresh),
            status=status,
            confidence=confidence,
        )

        # logging
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), ear_smooth, iris_smooth, pose_smooth[0], pose_smooth[1], pose_smooth[2], status, confidence])
        except Exception:
            pass
        # Draw status with appropriate color
        if status.startswith("Focused"):
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        ih, iw = bgr.shape[:2]
        font_scale = max(0.6, min(3.0, ih / 480.0))
        thickness = max(1, int(round(ih / 240.0)))
        cv2.putText(bgr, f"Status: {status}", (30, int(30 * (ih / 480.0))), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        if return_metrics:
            return bgr, metrics
        return bgr
    else:
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        ih, iw = bgr.shape[:2]
        font_scale = max(0.6, min(3.0, ih / 480.0))
        thickness = max(1, int(round(ih / 240.0)))
        cv2.putText(bgr, "No face detected", (30, int(30 * (ih / 480.0))), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness)
        if return_metrics:
            return bgr, dict(status="no_face")
        return bgr

use_blocks = False
try:
    _ = gr.Blocks
except Exception:
    use_blocks = False

if use_blocks:
    with gr.Blocks(title="Focus Detector") as demo:
        gr.Markdown("# 👁️ Focus Detector")
        gr.Markdown("Detects if you are **FOCUSED** (looking at screen) or **NOT FOCUSED** (looking away/distracted)")
        
        with gr.Tabs():
            # WEBCAM TAB
            with gr.TabItem("📹 Live Webcam"):
                gr.Markdown("### Real-time Detection")
                gr.Markdown("The webcam feed shows **live analysis** - your status updates continuously")
                
                with gr.Row():
                    with gr.Column():
                        cam = gr.Image(sources=["webcam"], tool=None, label="📷 Your Webcam", type="numpy")
                    with gr.Column():
                        out_cam = gr.Image(label="✅ Detection Output")
                
                cam.change(fn=detect_focus, inputs=cam, outputs=out_cam)
            
            # UPLOAD TAB
            with gr.TabItem("📤 Upload Image"):
                gr.Markdown("### Test with Saved Images")
                gr.Markdown("Upload a photo to analyze. The system will detect if the person in the image is **focused** (looking at camera) or **not focused** (looking away)")
                
                with gr.Row():
                    with gr.Column():
                        upload = gr.Image(type="numpy", label="📸 Upload Image")
                    with gr.Column():
                        out_up = gr.Image(label="✅ Detection Output")
                
                with gr.Row():
                    out_metrics = gr.Textbox(label="📊 Analysis Metrics", lines=12)

                def _process_upload(img):
                    res = detect_focus(img, return_metrics=True)
                    if isinstance(res, tuple):
                        img_out, metrics = res
                        import json
                        return img_out, json.dumps(metrics, indent=2)
                    return res, "{}"

                upload.change(fn=_process_upload, inputs=upload, outputs=[out_up, out_metrics])


    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)
else:
    img_input = gr.Image(type="numpy")
    outputs = [gr.Image(), gr.Textbox(lines=10, label="Metrics")]

    def _iface_process(img):
        res = detect_focus(img, return_metrics=True)
        if isinstance(res, tuple):
            img_out, metrics = res
            import json
            return img_out, json.dumps(metrics, indent=2)
        return res, "{}"

    demo = gr.Interface(
        fn=_iface_process,
        inputs=img_input,
        outputs=outputs,
        live=False,
        title="Focus Detector",
        description="Upload an image to detect focus or sleepy state."
    )

    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)
