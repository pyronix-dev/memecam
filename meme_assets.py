"""Which meme is glued to which gesture, and how it is anchored to the face.

anchor:
  "face"  -> centred on the face box, scaled to COVER it on both axes
             (scale is the margin factor: 1.0 = exactly the face box)
  "above" -> same size rule, but sitting on top of the head
  "side"  -> parked beside your head on the roomier side, width = scale * face_width
  "eyes"  -> aligned + rotated to the eye line, width = scale * eye_distance
dx / dy are offsets: face/above in face-box units, eyes in eye-distance units.

Every entry is a GIF pulled from Giphy by `fetch_gifs.py`.
Table order = the force-key order in the app.
"""
from __future__ import annotations

import os

from overlay_fx import find_asset, load_sprite

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

GESTURE_STYLE = {
    "HEART":        dict(file="heart",         anchor="face", scale=1.35, dx=0.0, dy=-0.05),
    "TIMEOUT":      dict(file="timeout",       anchor="face", scale=1.35, dx=0.0, dy=0.0),
    "SCUBA":        dict(file="scuba",         anchor="face", scale=1.35, dx=0.0, dy=-0.05),
    "MIND_BLOWN":   dict(file="mind_blown",    anchor="face", scale=1.40, dx=0.0, dy=-0.05),
    "TONGUE":       dict(file="tongue",        anchor="face", scale=1.30, dx=0.0, dy=-0.05),
    "HORSE":        dict(file="horse",         anchor="face", scale=1.35, dx=0.0, dy=-0.05),
    "HANDS_MOUTH":  dict(file="hands_mouth",   anchor="face", scale=1.35, dx=0.0, dy=-0.05),
    "THUMBS_UP":    dict(file="thumbs_up",     anchor="side", scale=1.30, dx=0.0, dy=0.05),
    "DOUBLE_THUMBS": dict(file="double_thumbs", anchor="face", scale=1.40, dx=0.0, dy=-0.05),
    "WINK":         dict(file="wink",          anchor="face", scale=1.35, dx=0.0, dy=-0.05),
    "CALL_ME":      dict(file="call_me",       anchor="face", scale=1.35, dx=0.0, dy=-0.05),
    "SIDE_EYE":     dict(file="side_eye",      anchor="face", scale=1.35, dx=0.0, dy=-0.05),
}

GESTURE_HOWTO = {
    "HEART":        "two-hand heart (index tips + thumb tips touching)",
    "TIMEOUT":      "time-out 'T' (one hand flat, other vertical under it)",
    "SCUBA":        "pinch your nose shut (thumb + index on the nose)",
    "MIND_BLOWN":   "both open hands up beside/above your head, fingers spread",
    "TONGUE":       "stick your tongue out",
    "HORSE":        "stare: eyes wide, no hands in shot, hold it",
    "HANDS_MOUTH":  "both hands clapped over your MOUTH",
    "THUMBS_UP":    "one thumbs up, held away from your face",
    "DOUBLE_THUMBS": "BOTH thumbs up",
    "WINK":         "wink - one eye shut, the other open, hold it",
    "CALL_ME":      "'call me' hand: thumb + pinky out, middle fingers folded",
    "SIDE_EYE":     "side eye - cut your eyes hard to one side without turning",
}


def load_all(asset_dir: str = ASSET_DIR, verbose: bool = True):
    """Return {gesture: Sprite} for every asset that exists on disk."""
    sprites = {}
    for gesture, style in GESTURE_STYLE.items():
        path = find_asset(asset_dir, style["file"])
        if path is None:
            if verbose:
                print(f"  [skip] {gesture:<12} no assets/{style['file']}.(gif|webp|png)")
            continue
        try:
            sprites[gesture] = load_sprite(path)
        except Exception as exc:                      # a corrupt download shouldn't kill the app
            if verbose:
                print(f"  [fail] {gesture:<12} {os.path.basename(path)}: {exc}")
            continue
        if verbose:
            spr = sprites[gesture]
            print(f"  [ ok ] {gesture:<12} {os.path.basename(path):<22} "
                  f"{spr.n_frames:>3} frame(s)")
    return sprites
