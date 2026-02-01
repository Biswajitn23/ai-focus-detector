import os
import argparse
import glob
import cv2
import numpy as np
USE_TASKS = False
FaceLandmarker = None
FaceLandmarkerOptions = None
_BaseOptions = None
Image = None
ImageFormat = None
mps = None

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
        USE_TASKS = False
    except Exception as e:
        raise ImportError("Neither MediaPipe Tasks nor Solutions FaceMesh are available") from e
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Feature extraction using MediaPipe FaceMesh landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FACE_2D_IDX = [1, 152, 263, 33, 287, 57, 61, 291, 199]


def eye_aspect_ratio(landmarks, eye_indices):
    points = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    vertical1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    vertical2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (vertical1 + vertical2) / (2.0 * horizontal)


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
    image_points = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_2D_IDX],
        dtype="double",
    )
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0, 0, 0
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    x = np.arctan2(rmat[2, 1], rmat[2, 2])
    y = np.arctan2(-rmat[2, 0], sy)
    z = np.arctan2(rmat[1, 0], rmat[0, 0])
    return np.degrees(x), np.degrees(y), np.degrees(z)


def extract_features(image_path, face_mesh=None, face_landmarker=None):
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if face_landmarker is not None:
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
    else:
        res = face_mesh.process(rgb)
        if not res or not res.multi_face_landmarks:
            return None
        landmarks = res.multi_face_landmarks[0].landmark
    
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    left_iris = iris_aspect_ratio(landmarks, LEFT_IRIS)
    right_iris = iris_aspect_ratio(landmarks, RIGHT_IRIS)
    ear = (left_ear + right_ear) / 2.0
    iris = (left_iris + right_iris) / 2.0
    h, w, _ = rgb.shape
    yaw, pitch, roll = head_pose(landmarks, w, h)
    
    # Gaze: normalized iris offset per eye (indicates looking left/right/center)
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
    
    return np.array([ear, iris, yaw, pitch, roll, gaze_x], dtype=np.float32)


def load_dataset(data_dir):
    X, y = [], []
    for label, subdir in [(1, "focused"), (0, "not_focused")]:
        pattern = os.path.join(data_dir, subdir, "**", "*.jp*g")
        paths = glob.glob(pattern, recursive=True)
        yield_paths = paths
        for p in yield_paths:
            yield p, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder with focused/ and not_focused/ subfolders")
    parser.add_argument("--output", default="focus_model.pkl", help="Output model path")
    args = parser.parse_args()

    face_mesh = None
    face_landmarker = None
    if USE_TASKS:
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker_v2.task")
        options_init = FaceLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        face_landmarker = FaceLandmarker.create_from_options(options_init)
    else:
        face_mesh = mps.face_mesh.FaceMesh(
            static_image_mode=True, refine_landmarks=True, max_num_faces=1
        )

    X, y = [], []
    total = 0
    used = 0
    focused_used = 0
    not_focused_used = 0
    skipped = []
    for path, label in load_dataset(args.data_dir):
        total += 1
        feats = extract_features(path, face_mesh=face_mesh, face_landmarker=face_landmarker)
        if feats is None:
            skipped.append(path)
            continue
        X.append(feats)
        y.append(label)
        used += 1
        if label == 1:
            focused_used += 1
        else:
            not_focused_used += 1

    print(f"Total images: {total}, Used: {used}, Focused: {focused_used}, Not focused: {not_focused_used}")
    if skipped:
        print("Skipped (no face detected):")
        for p in skipped:
            print(f"  - {p}")

    if used < 2 or focused_used == 0 or not_focused_used == 0:
        raise RuntimeError("Not enough usable samples per class. Add clearer face images to both focused/ and not_focused/.")

    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)

    if len(X) < 10:
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) if X_test is not None else None

    joblib.dump(clf, args.output)
    print(f"Saved model to {args.output}")
    if acc is None:
        print("Test accuracy: N/A (dataset too small; trained on all samples)")
    else:
        print(f"Test accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
