"""
STEP 2 — DETECT FACES
======================
Reads frames from processed_dataset/{real,fake}/
Detects the biggest face with MTCNN, saves 224×224 crop.
Output → processed_faces/{real,fake}/

Run:  python step2_detect_faces.py
"""

import os
import cv2
from mtcnn import MTCNN

# ── CONFIG ────────────────────────────────────────────────
REAL_INPUT  = "processed_dataset/real"
FAKE_INPUT  = "processed_dataset/fake"
REAL_OUTPUT = "processed_faces/real"
FAKE_OUTPUT = "processed_faces/fake"
# ──────────────────────────────────────────────────────────

os.makedirs(REAL_OUTPUT, exist_ok=True)
os.makedirs(FAKE_OUTPUT, exist_ok=True)

detector = MTCNN()


def detect_faces(input_folder, output_folder, label):
    video_dirs = [d for d in os.listdir(input_folder)
                  if os.path.isdir(os.path.join(input_folder, d))]
    print(f"\nTotal {label} video folders: {len(video_dirs)}")
    total = 0

    for folder_name in video_dirs:
        src_dir  = os.path.join(input_folder,  folder_name)
        dst_dir  = os.path.join(output_folder, folder_name)
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
                continue

            # biggest face
            bx, by, bw, bh = max(faces, key=lambda f: f['box'][2] * f['box'][3])['box']
            bx, by = max(0, bx), max(0, by)
            crop   = img[by:by+bh, bx:bx+bw]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (224, 224))
            cv2.imwrite(os.path.join(dst_dir, img_name), crop)
            saved += 1
            total += 1

        print(f"  {folder_name}: {saved} faces saved")

    print(f"Total {label} faces saved: {total}")


detect_faces(REAL_INPUT,  REAL_OUTPUT, "REAL")
detect_faces(FAKE_INPUT,  FAKE_OUTPUT, "FAKE")
print("\nStep 2 complete.")
