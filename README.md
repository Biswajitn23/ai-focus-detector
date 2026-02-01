# AI Focus Detector
![Demo animation](assets/animation.svg)

**This repository contains a webcam-based focus detector using MediaPipe face landmarks and a trained machine learning classifier.**

## Files
- **`app.py`** — Gradio web interface for focus detection. Supports live webcam input and image upload. Uses trained `focus_model.pkl` with heuristic fallback.
- **`train_focus.py`** — Training pipeline for the focus classifier. Extracts 6 features from face landmarks and trains a scikit-learn LogisticRegression model.
- **`download_datasets.py`** — Automated dataset downloader using Bing Image Search. Collects focused/not_focused training samples.
- **`focus_app.py`** — Streamlit web app with manual calibration and status display.
- **`focus_detector.py`** — CLI/OpenCV app: fullscreen webcam window with keyboard controls (`c` = calibrate, `q` = quit).
- **`focus_model.pkl`** — Trained scikit-learn Pipeline (StandardScaler + LogisticRegression). Classifies focus state from face features.

## Key Features
- **Machine Learning Classifier**: Trained LogisticRegression model on ~34 labeled images (25 focused, 9 not_focused)
- **6-Dimensional Feature Vector**:
  - **EAR** (Eye Aspect Ratio) - detects eye closure
  - **IAR** (Iris Aspect Ratio) - detects squinting
  - **Head Pose** (yaw, pitch, roll) - detects turned-away faces
  - **Gaze Direction** (iris offset) - detects looking left/right vs. looking at camera
- **Model Accuracy**: 71.4% on test set
- **Heuristic Fallback**: If model unavailable, uses eye closure + head pose heuristics
- **Temporal Smoothing**: Moving average over 10 frames for stable predictions
- **Auto-calibration**: Learns per-user eye thresholds in first few frames
- **Logging**: Per-frame analysis saved to `focus_log.csv`

## Quick Start

### 1. Setup
```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Web App
```powershell
python app.py
```
Open http://127.0.0.1:7861 in your browser

### 3. Train a Custom Model (Optional)
Prepare training data in `data/` folder:
```
data/
├── focused/        (images of people looking at camera/screen)
│   ├── image1.jpg
│   └── image2.jpg
└── not_focused/    (images of people looking away/distracted)
    ├── image1.jpg
    └── image2.jpg
```

Then train:
```powershell
python train_focus.py --data_dir data
```

### 4. Auto-Download Training Data (Optional)
```powershell
python download_datasets.py
```

## App Modes

### **Upload Tab** - Test with static images
- Upload a JPG/PNG image
- System analyzes the person's focus state
- Shows confidence score and detailed metrics (EAR, head pose, gaze direction)

### **Live Webcam Tab** - Real-time detection
- Uses webcam for continuous analysis
- Updates detection every frame
- Shows running status

## Model Details

### Training Pipeline
The `train_focus.py` script:
1. Loads images from `data/focused/` and `data/not_focused/`
2. Extracts 6 features per face using MediaPipe FaceMesh
3. Handles class imbalance with `class_weight='balanced'`
4. Trains LogisticRegression with StandardScaler normalization
5. Saves to `focus_model.pkl` for inference

### Feature Extraction
```python
features = [
    ear_smooth,      # Smoothed Eye Aspect Ratio
    iris_smooth,     # Smoothed Iris Aspect Ratio
    yaw,             # Head rotation left/right (degrees)
    pitch,           # Head rotation up/down (degrees)
    roll,            # Head tilt (degrees)
    gaze_x           # Normalized iris offset (indicates looking direction)
]
```

**Key Insight**: Gaze direction (how far the iris is from center) is the strongest predictor of focus. When looking at a camera/screen (focused), `gaze_x ≈ 0`. When looking to the side (not focused), `gaze_x` is large.

## Troubleshooting

### "side.jpeg uploading showing focused" 
**Fixed in latest version**: Missing gaze_x feature has been added to model prediction. The model now correctly detects side profiles as "Not Focused" due to high gaze offset.

### Webcam not working
- Ensure no other app is using the camera
- Grant browser permission to access webcam
- Try uploading an image instead to verify the model works

### Model shows old sklearn version warning
- The model was trained with scikit-learn 1.3.2 but you have 1.8.0+
- This is safe to ignore; the model still works correctly

### Low accuracy on your images
- Try retraining with more labeled samples (especially "not_focused" images)
- Current model has class imbalance (25 focused vs 9 not_focused)
- Use `download_datasets.py` to auto-collect more training data

## Deployment to Hugging Face Spaces

```powershell
copy requirements.txt requirements.txt
git add requirements.txt
git commit -m "Update requirements"
huggingface-cli login
gradio deploy
```

## Next Steps
- Collect more "not_focused" training samples to improve model balance
- Fine-tune confidence thresholds for your specific use case
- Add blink rate and pupil dilation as additional features
- Implement batch processing for analyzing multiple images at once
