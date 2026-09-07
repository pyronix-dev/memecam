"""Sprite loading (PNG / animated GIF / animated WebP) and alpha compositing."""
from __future__ import annotations

import math
import os
import glob

import cv2
import numpy as np
from PIL import Image, ImageSequence

MAX_SOURCE_WIDTH = 720            # pre-shrink huge meme gifs once, at load time
WIDTH_STEP = 8                    # quantise overlay sizes so the resize cache hits
CACHE_BUDGET_BYTES = 96 << 20     # per-sprite resize cache ceiling


# --------------------------------------------------------------------------- #
# Alpha compositing
# --------------------------------------------------------------------------- #
def alpha_blend(dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int) -> None:
    """Blend a BGRA sprite onto a BGR frame at (x, y), in place, with clipping.

    This is the correct way to honour a PNG/GIF alpha channel in OpenCV:
    out = src_rgb * a + dst_rgb * (1 - a).  Simply doing dst[...] = src[..., :3]
    is what makes transparent areas show up as solid black.
    """
    if src_bgra is None or src_bgra.size == 0:
        return
    sh, sw = src_bgra.shape[:2]
    dh, dw = dst_bgr.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, dw), min(y + sh, dh)
    if x0 >= x1 or y0 >= y1:
        return  # fully off-screen

    src = src_bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = dst_bgr[y0:y1, x0:x1]

    alpha = src[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, None]
    np.copyto(roi, (src[:, :, :3].astype(np.float32) * alpha
                    + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8))


def rotate_rgba(img_bgra: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a BGRA sprite around its centre, expanding the canvas so nothing clips."""
    if abs(degrees) < 0.75:
        return img_bgra
    h, w = img_bgra.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(img_bgra, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


# --------------------------------------------------------------------------- #
# Sprites
# --------------------------------------------------------------------------- #
class Sprite:
    """An animated (or single-frame) BGRA image with a resize cache."""

    def __init__(self, frames, durations_ms, name="sprite"):
        self.name = name
        self.frames = frames
        self.durations = np.cumsum(np.asarray(durations_ms, dtype=np.float64))
        self.total_ms = float(self.durations[-1])
        self._cache: dict = {}
        self._cache_bytes = 0

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def _index_at(self, elapsed_s: float) -> int:
        if self.n_frames == 1 or self.total_ms <= 0:
            return 0
        t = (elapsed_s * 1000.0) % self.total_ms
        return int(min(np.searchsorted(self.durations, t, side="right"), self.n_frames - 1))

    def get(self, elapsed_s: float, target_w: int) -> np.ndarray:
        """Return the animation frame for `elapsed_s`, scaled to `target_w` px wide.

        The width is quantised to WIDTH_STEP px: the face box wobbles by a pixel or
        two every frame, and without this every frame would be a cache miss (and a
        fresh resize) for a width nobody can tell apart."""
        target_w = int(max(8, min(target_w, 4000)))
        target_w -= target_w % WIDTH_STEP
        idx = self._index_at(elapsed_s)
        key = (idx, target_w)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        src = self.frames[idx]
        h, w = src.shape[:2]
        target_h = max(8, int(round(h * target_w / float(w))))
        interp = cv2.INTER_AREA if target_w < w else cv2.INTER_LINEAR
        out = cv2.resize(src, (target_w, target_h), interpolation=interp)

        self._cache_bytes += out.nbytes      # bounded by bytes: GIFs get big
        if self._cache_bytes > CACHE_BUDGET_BYTES:
            self._cache.clear()
            self._cache_bytes = out.nbytes
        self._cache[key] = out
        return out


def _pil_to_bgra(frame: Image.Image) -> np.ndarray:
    rgba = np.array(frame.convert("RGBA"))
    return np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])


def load_sprite(path: str) -> Sprite:
    """Load PNG / GIF / WebP (animated or not) into a Sprite of BGRA frames."""
    img = Image.open(path)
    frames, durations = [], []
    for frame in ImageSequence.Iterator(img):
        bgra = _pil_to_bgra(frame)
        h, w = bgra.shape[:2]
        if w > MAX_SOURCE_WIDTH:             # keep per-frame resize cheap
            scale = MAX_SOURCE_WIDTH / float(w)
            bgra = cv2.resize(bgra, (MAX_SOURCE_WIDTH, max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
        frames.append(bgra)
        durations.append(max(20, int(frame.info.get("duration", 100))))
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return Sprite(frames, durations, name=os.path.basename(path))


def find_asset(asset_dir: str, stem: str, ext_priority=(".gif", ".webp", ".apng", ".png")):
    """Find assets/<stem>.<ext>, preferring animated formats, so swapping in a new
    meme is just a matter of dropping the file next to the old one."""
    for ext in ext_priority:
        for path in sorted(glob.glob(os.path.join(asset_dir, stem + ext))):
            return path
    return None
