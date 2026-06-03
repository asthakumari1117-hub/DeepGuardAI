"""
STEP 1 — EXTRACT FRAMES
========================
Reads videos from:
  videos/real/   → 16 uniform frames → processed_dataset/real/<video_name>/
  videos/fake/   → 16 uniform frames → processed_dataset/fake/<video_name>/

Run:  python step1_extract_frames.py
"""

import os
import cv2
import numpy as np

# ── CONFIG ────────────────────────────────────────────────
REAL_FOLDER   = "videos/real"
FAKE_FOLDER   = "videos/fake"
REAL_OUTPUT   = "processed_dataset/real"
FAKE_OUTPUT   = "processed_dataset/fake"
SEQUENCE_LEN  = 16
IMAGE_SIZE    = 224
# ──────────────────────────────────────────────────────────

os.makedirs(REAL_OUTPUT, exist_ok=True)
os.makedirs(FAKE_OUTPUT, exist_ok=True)


def extract_uniform_frames(video_path):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = (
        np.linspace(0, total - 1, SEQUENCE_LEN, dtype=int)
        if total >= SEQUENCE_LEN
        else np.arange(total)
    )
    index_set = set(indices.tolist())

    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in index_set:
            frames.append(cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)))
        idx += 1
    cap.release()

    if not frames:
        return []
    while len(frames) < SEQUENCE_LEN:        # pad short videos
        frames.append(frames[-1])
    return frames[:SEQUENCE_LEN]


def process_videos(video_folder, output_folder, label):
    files = os.listdir(video_folder)
    print(f"\nTotal {label} videos: {len(files)}")
    ok = 0
    for name in files:
        if not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        path   = os.path.join(video_folder, name)
        frames = extract_uniform_frames(path)
        if len(frames) != SEQUENCE_LEN:
            print(f"  SKIP {name} (only {len(frames)} frames)")
            continue
        save_dir = os.path.join(output_folder, os.path.splitext(name)[0])
        os.makedirs(save_dir, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(save_dir, f"{i:04d}.jpg"), f)
        print(f"  OK   {name}")
        ok += 1
    print(f"Saved {ok} {label} video sequences.")


process_videos(REAL_FOLDER, REAL_OUTPUT, "REAL")
process_videos(FAKE_FOLDER, FAKE_OUTPUT, "FAKE")
print("\nStep 1 complete.")
