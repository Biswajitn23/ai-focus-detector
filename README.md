# AI Focus Detector
![Demo animation](assets/animation.svg)
**This repository contains three entrypoints for a webcam-based focus / drowsiness detector that uses MediaPipe landmarks.**

## Files
- **`app.py`** — Gradio web app (live webcam demo). Prefers MediaPipe Tasks `FaceLandmarker`; falls back to `mediapipe.solutions.face_mesh` if needed.
- **`focus_app.py`** — Streamlit web app with manual calibration and status display.
- **`focus_detector.py`** — CLI/OpenCV app: fullscreen webcam window with keyboard controls (`c` = calibrate, `q` = quit).

## Key features (present across all three apps)
- **EAR + IAR**: Eye Aspect Ratio and Iris Aspect Ratio per-frame detection.
- **Head-pose gating** via solvePnP to detect turned-away faces.
- **Temporal smoothing & debounce**: moving average and consecutive-frame thresholds.
- **Auto-calibration**: collect per-user baselines.
- **API fallback**: Tasks API -> Solutions FaceMesh.
- **Logging**: per-frame logs to `focus_log.csv` for offline analysis.

## Quick start (local)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
```

2. Install dependencies (use the tested pinned file):

```powershell
pip install -r requirements_new.txt
```

3. Run an app:
- Gradio (web demo):

```powershell
python app.py
```

- Streamlit (browser UI):

```powershell
streamlit run focus_app.py
```

- CLI (no browser):

```powershell
python focus_detector.py
```

## Calibration & Controls
- **`focus_detector.py`**: press `c` to calibrate, `q` to quit.
- **`focus_app.py`**: use the **Calibrate** button in the Streamlit UI.
- **`app.py`**: uses auto-calibration by default (can be adapted for manual control).

## Deploy to Hugging Face Spaces
- Ensure a valid `requirements.txt` exists at repo root. Use `requirements_new.txt` as the tested set (copy before deploying):

```powershell
copy requirements_new.txt requirements.txt
git add requirements.txt
git commit -m "Pin requirements for HF Spaces"
```

- Deploy using Gradio CLI (must be logged in with a Hugging Face token):

```powershell
pip install --upgrade gradio huggingface_hub
huggingface-cli login
gradio deploy
```

## Notes & caveats
- MediaPipe builds can fail on some hosted platforms; the code falls back to `mediapipe.solutions.face_mesh` which often has prebuilt wheels.
- Large binary assets are better downloaded at runtime or added as large-file assets rather than committed to the repo.
- For best accuracy across people and lighting, use calibration, stronger temporal smoothing, and labeled logs (`focus_log.csv`) to train a classifier.

## Troubleshooting
- If webcam frames are empty: ensure no other app is using the camera and the OS/browser grants permission.
- If `mediapipe` import fails: try `pip install 'mediapipe>=0.10.9'` or rely on the fallback.
- If HF deploy fails due to large or binary files, create a clean deploy directory containing only the required files.

## Contact / Next steps
- Want me to copy `requirements_new.txt` → `requirements.txt` and push? Or add a shared `focus_core.py` to reduce duplication? Tell me which and I'll proceed.
