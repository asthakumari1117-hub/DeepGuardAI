# DeepGuard AI 🛡️

An AI-powered Deepfake Detection System built using Python, PyTorch, OpenCV, ResNet18, Bidirectional LSTM, and Streamlit.

## Features

* Deepfake Video Detection
* Face Detection using MTCNN
* Face Alignment
* ResNet18 Feature Extraction
* Bidirectional LSTM Classification
* Confidence Score
* REAL / FAKE Prediction
* Streamlit Web Interface
* Login Authentication System
* Video Metadata Analysis
* Explainable AI Support

---

## One-Time Setup

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Place videos inside:

```text
videos/
│
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│
└── fake/
    ├── fake1.mp4
    ├── fake2.mp4
```

---

## Training Pipeline

Run the following steps in order:

```bash
# Step 1 — Extract Frames
python step1_extract_frames.py

# Step 2 — Detect Faces
python step2_detect_faces.py

# Step 3 — Face Alignment
python step3_face_alignment.py

# Step 4 — Build LSTM Dataset
python step4_build_dataset.py

# Step 5 — Train Model
python step5_train.py
```

---

## Run DeepGuard AI

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Default Login

```text
Username: admin
Password: admin
```

---

## Prediction Output

After uploading a video, the system provides:

* REAL / FAKE Result
* Confidence Score
* Video Resolution
* FPS
* Duration
* Frame Count
* File Size
* Video Quality Information
* Processing Steps Visualization

---

## Project Structure

```text
DeepGuardAI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── step1_extract_frames.py
├── step2_detect_faces.py
├── step3_face_alignment.py
├── step4_build_dataset.py
├── step5_train.py
│
├── videos/
├── extracted_frames/
├── detected_faces/
├── aligned_faces/
├── dataset_features/
│
└── best_lstm_model.pth
```

---

## Model Architecture

* ResNet18 Feature Extractor
* Bidirectional LSTM
* Batch Normalization
* Dropout Regularization
* Softmax Classification

---

## Performance Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Confidence Score

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* Streamlit
* NumPy
* Pandas
* Scikit-Learn
* MTCNN

---

## Author

Astha Kumari

DeepGuard AI – Deepfake Detection using Deep Learning.

