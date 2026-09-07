#!/usr/bin/env python3
"""One-time download of the two MediaPipe Tasks model bundles (~9 MB total).

MediaPipe >= 0.10.3x ships no bundled models (the old mp.solutions API is gone),
so the Tasks API needs these .task files on disk.

    python fetch_models.py
"""
from __future__ import annotations

import os
import sys
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand_landmarker.task":
        f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task":
        f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    # Full-range, not short-range: the short-range detector (and the one bundled
    # inside face_landmarker) only sees faces wider than ~10% of the frame, so it
    # finds nobody once three or more people share the shot.
    "blaze_face_full_range.tflite":
        f"{BASE}/face_detector/blaze_face_full_range/float16/latest/"
        "blaze_face_full_range.tflite",
}


def ensure_models(model_dir: str = MODEL_DIR, quiet: bool = False) -> dict:
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        dest = os.path.join(model_dir, name)
        if not os.path.exists(dest) or os.path.getsize(dest) < 1024:
            if not quiet:
                print(f"  downloading {name} ...", flush=True)
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
        paths[name] = dest
        if not quiet:
            print(f"  [ ok ] {name:<28} {os.path.getsize(dest) / 1e6:5.1f} MB")
    return paths


if __name__ == "__main__":
    try:
        ensure_models()
    except Exception as exc:
        print(f"! download failed: {exc}")
        sys.exit(1)
    print(f"\nModels in {MODEL_DIR}")
