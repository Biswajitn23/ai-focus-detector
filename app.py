import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.face_landmarker import _BaseOptions
# Robust import for Image and ImageFormat with helpful error if not available
def _import_mediapipe_image():
    try:
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        return Image, ImageFormat
    except Exception:
        try:
            from mediapipe.tasks.python.vision import Image, ImageFormat
            return Image, ImageFormat
        except Exception as e:
            try:
                from importlib.metadata import version
                mp_version = version("mediapipe")
            except Exception:
                try:
                    import pkg_resources
                    mp_version = pkg_resources.get_distribution("mediapipe").version
                except Exception:
                    mp_version = "unknown"
            raise ImportError(
                f"Could not import Image/ImageFormat from MediaPipe tasks API. Installed mediapipe version: {mp_version}. "
                "This app requires MediaPipe Tasks (mediapipe>=0.10).\n"
                "Please install a compatible mediapipe, e.g. `pip install 'mediapipe>=0.10.0'`"
            ) from e

Image, ImageFormat = _import_mediapipe_image()
import os

MODEL_PATH = "face_landmarker_v2.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task"

# Download model if not
def download_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

download_model()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

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

def detect_focus(frame):
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Setup FaceLandmarker
    options = FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    result = face_landmarker.detect(mp_image)
    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        left_iris_ar = iris_aspect_ratio(landmarks, LEFT_IRIS)
        right_iris_ar = iris_aspect_ratio(landmarks, RIGHT_IRIS)
        ear = (left_ear + right_ear) / 2
        iris_ar = (left_iris_ar + right_iris_ar) / 2
        # Draw status
        if ear < 0.20 and iris_ar < 0.25:
            status = "Sleepy/Not Focused"
            color = (0, 0, 255)
        else:
            status = "Focused"
            color = (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame

demo = gr.Interface(
    fn=detect_focus,
    inputs=gr.Image(source="webcam", tool=None),
    outputs=gr.Image(),
    live=True,
    title="Focus Detector (Webcam)",
    description="Detects if you are focused or sleepy using your webcam."
)

demo.launch()
