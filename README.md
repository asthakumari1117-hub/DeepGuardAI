# DeepGuard AI — Setup & Run Guide

## One-time setup

```bash
pip install -r requirements.txt
```

## Training pipeline (run once)

Put your videos in:
```
videos/
  real/     ← authentic video files (.mp4 .avi .mov .mkv)
  fake/     ← deepfake video files
```

Then run each step in order:

```bash
# Step 1 — Extract 16 frames per video
python step1_extract_frames.py

# Step 2 — Detect and crop faces (MTCNN)
python step2_detect_faces.py

# Step 3 — Align faces by eye landmarks
python step3_face_alignment.py

# Step 4 — Build LSTM feature dataset  ← KEY FIX: reads aligned_faces/, not optical_flow/
python step4_build_dataset.py

# Step 5 — Train Bidirectional LSTM (saves best_lstm_model.pth)
python step5_train.py
```

## Launch the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.
Default login: **admin / admin**

## Upload a video → click Analyse → get REAL / FAKE result

---

## What was broken (and fixed)

| Bug | Old code | Fixed code |
|-----|----------|-----------|
| Wrong training input | `optical_flow/` images fed to ResNet18 (cosine-sim 0.999 → model can't learn) | `aligned_faces/` images (actual face crops) |
| Missing normalisation at inference | `transforms.ToTensor()` only | Full ImageNet normalise matches training |
| Architecture mismatch | No BatchNorm in app.py model | BatchNorm1d added to match checkpoint |
| Wrong threshold | 75% fake vote needed | 50% majority vote (standard) |
| Training collapse | Hard-coded class weights | Balanced from actual label counts |
| No early stopping | Fixed 20 epochs | Patience=7 early stopping + gradient clipping |
