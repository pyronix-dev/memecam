"""MediaPipe Tasks wrappers: hands, and faces for everyone in the frame.

MediaPipe >= 0.10.3x removed the old `mp.solutions.*` API, so this uses the Tasks
API.  Two things here are not the obvious arrangement, both for multi-person:

* Faces are found with the **full-range** BlazeFace detector and then landmarked
  one crop at a time.  Running FaceLandmarker straight on the frame uses its own
  short-range detector, which was measured to miss any face narrower than ~10% of
  the frame width - it found 2/2 people but 0/3, 0/4 and 0/5.  Detect-then-crop
  finds 5/5 at ~3.8 ms for the detector plus ~6.7 ms per face.
* There is no `tongueOut` blendshape, so the tongue is measured from pixels -
  see `tongue_score`.
"""
from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "2")        # hush MediaPipe's C++ logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gesture_rules import (CHEEK_LEFT, CHEEK_RIGHT, Face, Hand, INNER_LIP_LOOP,
                           LIP_INNER_LOW, LIP_INNER_UP)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL = "hand_landmarker.task"
FACE_MESH_MODEL = "face_landmarker.task"
FACE_DET_MODEL = "blaze_face_full_range.tflite"

FACE_CROP_PAD = 0.75         # grow each detected face box by this before landmarking
TONGUE_MIN_VALUE = 0.38      # tongue is at least this fraction as bright as cheek skin
TONGUE_MIN_SAT = 60          # ... and this saturated (teeth are bright but grey)
TONGUE_HUE = 16              # ... within this many HSV degrees of red, either way
TONGUE_MIN_GAP = 0.05        # lips must be parted by this fraction of the face height
TONGUE_ERODE = 0.18          # shrink the mouth mask by this much of its height
TONGUE_ROI_H = 64            # normalise the mouth ROI to this height first, so the
                             # erosion means the same thing near and far from the camera


def tongue_score(rgb: np.ndarray, pts: np.ndarray) -> float:
    """Fraction of the mouth opening that looks like tongue. 0 when shut.

    The lips are themselves bright saturated red, so a nearly-closed mouth whose
    inner-lip polygon is a one-pixel sliver would otherwise score ~1.0.  Hence the
    parted-lips gate and the eroded mask.
    """
    face_h = float(pts[:, 1].max() - pts[:, 1].min())
    if abs(pts[LIP_INNER_LOW][1] - pts[LIP_INNER_UP][1]) < TONGUE_MIN_GAP * face_h:
        return 0.0
    loop = np.round(pts[INNER_LIP_LOOP]).astype(np.int32)
    x0, y0 = loop.min(axis=0)
    x1, y1 = loop.max(axis=0)
    h, w = rgb.shape[:2]
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1) + 1, w), min(int(y1) + 1, h)
    if x1 - x0 < 6 or y1 - y0 < 4:
        return 0.0

    # Normalise the crop to a fixed height: three metres from the webcam the mouth
    # is a dozen pixels tall and a percentage-based erode rounds down to nothing.
    crop = rgb[y0:y1, x0:x1]
    scale = min(4.0, max(1.0, TONGUE_ROI_H / float(y1 - y0)))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    ch, cw = crop.shape[:2]

    mask = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(mask, [np.round((loop - [x0, y0]) * scale).astype(np.int32)], 255)
    k = int(TONGUE_ERODE * ch)
    if k >= 1:                                   # pull the mask off the lip edges
        mask = cv2.erode(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    n = int(cv2.countNonZero(mask))
    if n < 40:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # brightness reference: two small cheek patches (same lighting as the face)
    skin = []
    for idx in (CHEEK_LEFT, CHEEK_RIGHT):
        cx, cy = np.round(pts[idx]).astype(int)
        r = max(3, int(0.02 * (pts[:, 0].max() - pts[:, 0].min())))
        patch = rgb[max(cy - r, 0):min(cy + r, h), max(cx - r, 0):min(cx + r, w)]
        if patch.size:
            skin.append(float(cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)[:, :, 2].mean()))
    skin_v = max(np.mean(skin) if skin else 128.0, 1.0)

    tongue = ((mask > 0)
              & (val > TONGUE_MIN_VALUE * skin_v)
              & (sat > TONGUE_MIN_SAT)
              & ((hue < TONGUE_HUE) | (hue > 180 - TONGUE_HUE)))
    return float(np.count_nonzero(tongue)) / n


def _crop_square(rgb: np.ndarray, box, pad: float = FACE_CROP_PAD):
    """Padded square crop around a detection box, zero-filled past the edges."""
    h, w = rgb.shape[:2]
    cx = box.origin_x + box.width / 2.0
    cy = box.origin_y + box.height / 2.0
    side = max(box.width, box.height) * (1.0 + pad)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    sub = rgb[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
    if sub.size == 0:
        return None, 0, 0
    out = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    oy, ox = max(-y0, 0), max(-x0, 0)
    out[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
    return np.ascontiguousarray(out), x0, y0


class Detector:
    def __init__(self, model_dir: str = MODEL_DIR, max_people: int = 5,
                 det_conf: float = 0.5, track_conf: float = 0.5,
                 face_det_conf: float = 0.4):
        self.max_people = max(1, int(max_people))
        paths = {}
        for key, name in (("hand", HAND_MODEL), ("mesh", FACE_MESH_MODEL),
                          ("det", FACE_DET_MODEL)):
            paths[key] = os.path.join(model_dir, name)
            if not os.path.exists(paths[key]):
                raise FileNotFoundError(
                    f"missing model {name} - run:  python fetch_models.py")

        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=paths["hand"]),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2 * self.max_people,
                min_hand_detection_confidence=det_conf,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=track_conf))
        self.face_det = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=paths["det"]),
                running_mode=vision.RunningMode.VIDEO,
                # Looser than the hand threshold on purpose: the fifth face in a
                # group is small and scores lower, and a missed face means that
                # person simply cannot trigger a meme.
                min_detection_confidence=face_det_conf))
        self.mesh = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=paths["mesh"]),
                running_mode=vision.RunningMode.IMAGE,     # one crop at a time
                num_faces=1,
                output_face_blendshapes=True))
        # Separate counters: hands and faces run on different threads in memecam,
        # and MediaPipe requires strictly increasing timestamps per task instance.
        self._hand_ts = -1
        self._face_ts = -1

    # ---------------- hands ----------------
    def process_hands(self, rgb: np.ndarray, ts_ms: int, out_size=None):
        """All hands in the frame. ~29 ms/frame; memecam runs it on its own thread."""
        h, w = rgb.shape[:2]
        ow, oh = out_size or (w, h)
        self._hand_ts = max(int(ts_ms), self._hand_ts + 1)
        res = self.hands.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._hand_ts)
        hands = []
        for i, landmarks in enumerate(res.hand_landmarks or []):
            label = "?"
            if res.handedness and i < len(res.handedness):
                label = res.handedness[i][0].category_name
            hands.append(Hand(landmarks, ow, oh, label))
        return hands

    # ---------------- faces ----------------
    def process_faces(self, rgb: np.ndarray, ts_ms: int):
        """Every face in the frame, as smoothing vectors (see Face.vector_from_landmarks)."""
        h, w = rgb.shape[:2]
        self._face_ts = max(int(ts_ms), self._face_ts + 1)
        det = self.face_det.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._face_ts)

        boxes = sorted((d.bounding_box for d in det.detections or []),
                       key=lambda b: b.width * b.height, reverse=True)[:self.max_people]
        out = []
        for box in boxes:
            crop, ox, oy = _crop_square(rgb, box)
            if crop is None:
                continue
            res = self.mesh.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            if not res.face_landmarks:
                continue
            ch, cw = crop.shape[:2]
            pts = np.array([[lm.x * cw + ox, lm.y * ch + oy]
                            for lm in res.face_landmarks[0]], dtype=np.float32)
            blend = {}
            if res.face_blendshapes:
                blend = {c.category_name: c.score for c in res.face_blendshapes[0]}
            # the tongue test samples the full frame, not the crop
            out.append(Face.vector_from_landmarks(pts, blend, tongue_score(rgb, pts)))
        return out

    def process(self, rgb: np.ndarray, ts_ms: int, out_size=None):
        """Both stages, sequentially. Returns (hands, [face_vector, ...])."""
        return self.process_hands(rgb, ts_ms, out_size), self.process_faces(rgb, ts_ms)

    def close(self):
        self.hands.close()
        self.face_det.close()
        self.mesh.close()
