"""
STEP 4 — BUILD LSTM DATASET
============================
FIX: reads aligned_faces/ (NOT optical_flow/).
     Optical flow images fed into an ImageNet ResNet produce
     indistinguishable features (cosine-sim 0.999). Using the
     actual aligned face images gives meaningful separation.

Extracts ResNet18 features (512-d) from each face frame.
Saves lstm_features.npy  (N, 16, 512)
      lstm_labels.npy     (N,)        0=real, 1=fake

Run:  python step4_build_dataset.py
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models

# ── CONFIG ────────────────────────────────────────────────
REAL_FOLDER  = "aligned_faces/real"   # ← FIXED (was optical_flow/real)
FAKE_FOLDER  = "aligned_faces/fake"   # ← FIXED (was optical_flow/fake)
SEQUENCE_LEN = 16
# ──────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── ResNet18 feature extractor ─────────────────────────────
cnn = models.resnet18(pretrained=True)
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])  # → (B, 512, 1, 1)
cnn.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

all_sequences, all_labels = [], []


def process_videos(main_folder, label):
    dirs = sorted([d for d in os.listdir(main_folder)
                   if os.path.isdir(os.path.join(main_folder, d))])
    print(f"\nTotal {('REAL' if label==0 else 'FAKE')} folders: {len(dirs)}")
    ok = 0

    for folder_name in dirs:
        folder_path = os.path.join(main_folder, folder_name)
        img_files   = sorted([f for f in os.listdir(folder_path)
                               if f.lower().endswith((".jpg",".jpeg",".png"))])
        sequence = []

        for img_name in img_files:
            img = cv2.imread(os.path.join(folder_path, img_name))
            if img is None:
                continue
            try:
                img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(img)
                t    = transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = cnn(t).squeeze().cpu().numpy()   # (512,)
                sequence.append(feat)
            except Exception as e:
                print(f"  Warning: {img_name} — {e}")
                continue

        if not sequence:
            print(f"  SKIP {folder_name} (no frames)")
            continue

        # pad / trim to SEQUENCE_LEN
        while len(sequence) < SEQUENCE_LEN:
            sequence.append(sequence[-1])
        sequence = sequence[:SEQUENCE_LEN]

        all_sequences.append(sequence)
        all_labels.append(label)
        ok += 1

    print(f"  Sequences saved: {ok}")


process_videos(REAL_FOLDER, 0)
process_videos(FAKE_FOLDER, 1)

X = np.array(all_sequences, dtype=np.float32)
y = np.array(all_labels,    dtype=np.int64)

# shuffle
idx = np.arange(len(X))
np.random.shuffle(idx)
X, y = X[idx], y[idx]

print(f"\nDataset shape: X={X.shape}  y={y.shape}")
print(f"Real: {(y==0).sum()}   Fake: {(y==1).sum()}")

np.save("lstm_features.npy", X)
np.save("lstm_labels.npy",   y)
print("\nSaved lstm_features.npy and lstm_labels.npy")
print("Step 4 complete.")
