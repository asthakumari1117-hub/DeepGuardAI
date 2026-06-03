"""
STEP 3 — FACE ALIGNMENT
========================
Reads processed_faces/{real,fake}/
Aligns each face by eye keypoints.
Output → aligned_faces/{real,fake}/

Run:  python step3_face_alignment.py
"""

import os
import cv2
import numpy as np
from mtcnn import MTCNN

# ── CONFIG ────────────────────────────────────────────────
REAL_INPUT  = "processed_faces/real"
FAKE_INPUT  = "processed_faces/fake"
REAL_OUTPUT = "aligned_faces/real"
FAKE_OUTPUT = "aligned_faces/fake"
# ──────────────────────────────────────────────────────────

os.makedirs(REAL_OUTPUT, exist_ok=True)
os.makedirs(FAKE_OUTPUT, exist_ok=True)

detector = MTCNN()


def align_face(image, left_eye, right_eye):
    lx, ly = map(float, left_eye)
    rx, ry = map(float, right_eye)
    center = ((lx + rx) / 2, (ly + ry) / 2)
    angle  = np.degrees(np.arctan2(ry - ly, rx - lx))
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def process_faces(input_folder, output_folder, label):
    video_dirs = [d for d in os.listdir(input_folder)
                  if os.path.isdir(os.path.join(input_folder, d))]
    print(f"\nTotal {label} video folders: {len(video_dirs)}")
    total = 0

    for folder_name in video_dirs:
        src_dir = os.path.join(input_folder,  folder_name)
        dst_dir = os.path.join(output_folder, folder_name)
        os.makedirs(dst_dir, exist_ok=True)
        saved = 0

        for img_name in sorted(os.listdir(src_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(src_dir, img_name))
            if img is None:
                continue
            rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if not faces:
                # no eyes found — save original face (still useful)
                cv2.imwrite(os.path.join(dst_dir, img_name),
                            cv2.resize(img, (224, 224)))
                saved += 1
                total += 1
                continue

            kp    = faces[0]['keypoints']
            aligned = align_face(img, kp['left_eye'], kp['right_eye'])
            aligned = cv2.resize(aligned, (224, 224))
            cv2.imwrite(os.path.join(dst_dir, img_name), aligned)
            saved += 1
            total += 1

        print(f"  {folder_name}: {saved} aligned faces saved")

    print(f"Total {label} aligned: {total}")


process_faces(REAL_INPUT,  REAL_OUTPUT, "REAL")
process_faces(FAKE_INPUT,  FAKE_OUTPUT, "FAKE")
print("\nStep 3 complete.")
