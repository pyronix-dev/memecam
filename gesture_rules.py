"""Landmarks + expression signals -> gesture, plus temporal smoothing.

All distance thresholds are multiples of the detected FACE WIDTH, so they behave
the same whether you sit close to or far from the camera.  Tune them in THRESH;
run the app with --debug to watch the live values.
"""
from __future__ import annotations

import time

import numpy as np

# ---- MediaPipe Hands landmark ids ----
WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
PALM_IDS = [0, 5, 9, 13, 17]

# ---- MediaPipe Face Mesh landmark ids (478-point model, iris included) ----
IRIS_RIGHT, IRIS_LEFT = 468, 473          # image-right / image-left iris centres
NOSE_TIP = 1
LIP_INNER_UP, LIP_INNER_LOW = 13, 14
INNER_LIP_LOOP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                  324, 318, 402, 317, 14, 87, 178, 88, 95]
CHEEK_LEFT, CHEEK_RIGHT = 330, 101        # skin reference patches

THRESH = {
    "heart_index_gap":   0.30,   # index fingertips must nearly touch
    "heart_thumb_gap":   0.42,   # thumb tips must nearly touch
    "heart_wrists":      0.80,   # ... while the two wrists splay this far apart
    "timeout_touch":     0.60,   # vertical hand's tips -> horizontal hand's palm
    "axis":              0.62,   # how axis-aligned a hand must be for the "T"
    "mouth_palm":        0.65,   # hands-over-mouth: palm -> mouth distance
    "mouth_spread":      0.45,   # ... and how far off the mouth's centre line
    "blown_apart":       0.80,   # mind-blown: the two palms must be far apart
    "blown_lift":        0.35,   # ... and above face_top + this * face_h
    "raised_hands":      0.45,   # "both hands are up by the head" suppression zone
    "scuba_pinch":       0.34,   # thumb/index tips must be this close together
    "scuba_nose":        0.30,   # ... and this close to the nose
    "hand_clearance":    0.75,   # free-hand poses must be AWAY from the face
    "tongue":            0.35,   # fraction of the mouth opening that looks tongue-y
    "mouth_open":        0.10,   # jawOpen blendshape gate for the tongue test
    "stare_wide":        0.22,   # eyeWide blendshape for the horse stare
    "stare_jaw":         0.30,   # ... with the mouth closed
    "wink_gap":          0.40,   # one eye shut while the other stays open
    "wink_open":         0.28,   # ... the open eye must be at most this closed
    "side_eye":          0.45,   # how hard the eyes are cut to one side
}

# Some gestures should be harder to trigger by accident than others.
# A single misread frame should never be enough for these: THUMBS_UP used to fire
# on the way *into* MIND_BLOWN, when one rising hand momentarily reads as thumb-only,
# and a wink has to outlast an ordinary blink.
ON_FRAMES = {"HORSE": 12, "SIDE_EYE": 8, "WINK": 4, "TONGUE": 4,
             "THUMBS_UP": 4, "DOUBLE_THUMBS": 3, "CALL_ME": 3}

GESTURES = ("HEART", "HANDS_MOUTH", "TIMEOUT", "MIND_BLOWN", "DOUBLE_THUMBS",
            "SCUBA", "CALL_ME", "THUMBS_UP", "TONGUE", "WINK", "SIDE_EYE", "HORSE")


def _d(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


class Hand:
    """Pixel-space view of one MediaPipe hand."""

    __slots__ = ("pts", "wrist", "palm", "span", "ext", "n_ext", "dirv", "label")

    def __init__(self, landmarks, w: int, h: int, label: str = "?"):
        # landmarks: the 21 MediaPipe Tasks NormalizedLandmarks for one hand
        self.pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        self.wrist = self.pts[WRIST]
        self.palm = self.pts[PALM_IDS].mean(axis=0)
        self.span = max(_d(self.pts[9], self.pts[0]), 1e-3)
        self.label = label
        self.ext = self._extended()
        self.n_ext = int(sum(self.ext))
        v = self.pts[9] - self.pts[0]
        n = np.linalg.norm(v)
        self.dirv = (v / n) if n > 1e-6 else np.array([0.0, -1.0], dtype=np.float32)

    def _extended(self):
        """Rotation-invariant finger-extension test: a fingertip is 'extended'
        when it is meaningfully farther from the wrist than its own PIP joint."""
        p, wr = self.pts, self.pts[WRIST]
        out = [_d(p[4], p[17]) > _d(p[3], p[17]) * 1.12]          # thumb: sideways test
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            out.append(_d(p[tip], wr) > _d(p[pip], wr) * 1.15)
        return out

    @property
    def thumb_dir(self):
        v = self.pts[THUMB_TIP] - self.pts[2]
        n = np.linalg.norm(v)
        return (v / n) if n > 1e-6 else np.array([0.0, -1.0], dtype=np.float32)

    def fingers_str(self) -> str:
        return "".join("1" if e else "0" for e in self.ext)


class Face:
    """Smoothed face state: box, key points and expression signals.

    Built from a flat vector so FaceTracker can exponentially smooth every field,
    expressions included - that alone removes most single-frame flicker.
    """

    N = 18
    __slots__ = ("x", "y", "w", "h", "cx", "cy", "right_eye", "left_eye", "nose",
                 "mouth", "jaw_open", "eye_wide", "blink_l", "blink_r", "gaze",
                 "tongue", "score")

    def __init__(self, vec, score=1.0):
        (self.x, self.y, self.w, self.h,
         rex, rey, lex, ley, nx, ny, mx, my,
         self.jaw_open, self.eye_wide, self.blink_l, self.blink_r,
         self.gaze, self.tongue) = vec
        self.cx, self.cy = self.x + self.w / 2.0, self.y + self.h / 2.0
        self.right_eye = np.array([rex, rey], dtype=np.float32)
        self.left_eye = np.array([lex, ley], dtype=np.float32)
        self.nose = np.array([nx, ny], dtype=np.float32)
        self.mouth = np.array([mx, my], dtype=np.float32)
        self.score = score

    @property
    def center(self):
        return np.array([self.cx, self.cy], dtype=np.float32)

    @property
    def blink(self) -> float:
        return (self.blink_l + self.blink_r) / 2.0

    @property
    def eye_dist(self) -> float:
        return max(_d(self.left_eye, self.right_eye), 1.0)

    @staticmethod
    def vector_from_landmarks(pts: np.ndarray, blend: dict, tongue: float) -> np.ndarray:
        """pts: (478, 2) pixel landmarks; blend: {blendshape_name: score}."""
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        mouth = (pts[LIP_INNER_UP] + pts[LIP_INNER_LOW]) / 2.0
        return np.array([
            x0, y0, x1 - x0, y1 - y0,
            pts[IRIS_RIGHT][0], pts[IRIS_RIGHT][1],
            pts[IRIS_LEFT][0], pts[IRIS_LEFT][1],
            pts[NOSE_TIP][0], pts[NOSE_TIP][1],
            mouth[0], mouth[1],
            blend.get("jawOpen", 0.0),
            (blend.get("eyeWideLeft", 0.0) + blend.get("eyeWideRight", 0.0)) / 2.0,
            blend.get("eyeBlinkLeft", 0.0),
            blend.get("eyeBlinkRight", 0.0),
            # signed gaze: positive = both eyes cut to the subject's left
            ((blend.get("eyeLookOutLeft", 0.0) + blend.get("eyeLookInRight", 0.0))
             - (blend.get("eyeLookOutRight", 0.0) + blend.get("eyeLookInLeft", 0.0))) / 2.0,
            tongue,
        ], dtype=np.float32)


class FaceTracker:
    """Exponential smoothing + a short 'coast' window, so overlays never jitter
    or pop off when a single frame misses the face."""

    def __init__(self, alpha=0.45, coast_s=0.5):
        self.alpha, self.coast_s = alpha, coast_s
        self._vec = None
        self._score = 0.0
        self._t = 0.0

    def update(self, vec, score, now):
        if vec is not None:
            self._vec = vec if self._vec is None else \
                self.alpha * vec + (1.0 - self.alpha) * self._vec
            self._score, self._t = score, now
        elif self._vec is not None and now - self._t > self.coast_s:
            self._vec = None
        return Face(self._vec, self._score) if self._vec is not None else None


# --------------------------------------------------------------------------- #
# Two-hand rules
# --------------------------------------------------------------------------- #
def _heart(a: Hand, b: Hand, f: Face, fw: float) -> bool:
    """Two-hand heart: index tips touch on top, thumb tips touch underneath, and
    the wrists splay apart.  Deliberately does NOT require an 'extended' index -
    in a real heart the index is curved, so its tip is no farther from the wrist
    than its own knuckle and the extension test fails."""
    idx_y = (a.pts[INDEX_TIP][1] + b.pts[INDEX_TIP][1]) / 2.0
    thumb_y = (a.pts[THUMB_TIP][1] + b.pts[THUMB_TIP][1]) / 2.0
    return (_d(a.pts[INDEX_TIP], b.pts[INDEX_TIP]) < THRESH["heart_index_gap"] * fw
            and _d(a.pts[THUMB_TIP], b.pts[THUMB_TIP]) < THRESH["heart_thumb_gap"] * fw
            and idx_y < thumb_y - 0.04 * fw
            and _d(a.wrist, b.wrist) > THRESH["heart_wrists"] * fw)


def _timeout(a: Hand, b: Hand, f: Face, fw: float) -> bool:
    """'T': one flat hand horizontal, the other vertical, tips touching its palm."""
    for vert, horiz in ((a, b), (b, a)):
        if (abs(horiz.dirv[0]) > THRESH["axis"] and abs(vert.dirv[1]) > THRESH["axis"]
                and vert.n_ext >= 3 and horiz.n_ext >= 3
                and _d(vert.pts[MIDDLE_TIP], horiz.palm) < THRESH["timeout_touch"] * fw):
            return True
    return False


def _mind_blown(a: Hand, b: Hand, f: Face, fw: float) -> bool:
    """Both open hands thrown up beside/above the head, wide apart."""
    top = f.y + THRESH["blown_lift"] * f.h
    return (a.n_ext >= 3 and b.n_ext >= 3
            and a.palm[1] < top and b.palm[1] < top
            and _d(a.palm, b.palm) > THRESH["blown_apart"] * fw)


def _double_thumbs(a: Hand, b: Hand, f: Face, fw: float) -> bool:
    """Two thumbs up, one per hand."""
    return all(_thumb_only(h) for h in (a, b))


def _hands_mouth(a: Hand, b: Hand, f: Face, fw: float) -> bool:
    """Both hands clapped over the mouth: close to it, and on its centre line, so
    hands held out by the cheeks or down at the chest do not count."""
    return all(_d(h.palm, f.mouth) < THRESH["mouth_palm"] * fw
               and abs(h.palm[0] - f.mouth[0]) < THRESH["mouth_spread"] * fw
               and h.palm[1] > f.cy - 0.10 * f.h for h in (a, b))


# --------------------------------------------------------------------------- #
# Single-hand rules
# --------------------------------------------------------------------------- #
def _scuba(h: Hand, f: Face, fw: float) -> bool:
    """Pinching the nose shut: thumb + index tips together, on the nose."""
    return (_d(h.pts[THUMB_TIP], h.pts[INDEX_TIP]) < THRESH["scuba_pinch"] * fw
            and _d(h.pts[INDEX_TIP], f.nose) < THRESH["scuba_nose"] * fw
            and _d(h.pts[THUMB_TIP], f.nose) < THRESH["scuba_nose"] * fw)


def _thumb_only(h: Hand) -> bool:
    return h.ext[0] and not any(h.ext[1:]) and h.thumb_dir[1] < -0.55


def _thumbs_up(h: Hand, f: Face, fw: float) -> bool:
    return _thumb_only(h) and _d(h.palm, f.center) > THRESH["hand_clearance"] * fw


def _call_me(h: Hand, f: Face, fw: float) -> bool:
    """Phone hand: thumb and pinky out, the three middle fingers folded."""
    return h.ext[0] and h.ext[4] and not h.ext[1] and not h.ext[2] and not h.ext[3]


# --------------------------------------------------------------------------- #
# Expression rules (no hands involved)
# --------------------------------------------------------------------------- #
def _tongue(f: Face) -> bool:
    return f.jaw_open > THRESH["mouth_open"] and f.tongue > THRESH["tongue"]


def _wink(f: Face) -> bool:
    """One eye shut, the other open. A normal blink shuts both, so it is the
    *difference* that matters, not how closed either eye is."""
    return (abs(f.blink_l - f.blink_r) > THRESH["wink_gap"]
            and min(f.blink_l, f.blink_r) < THRESH["wink_open"])


def _side_eye(f: Face) -> bool:
    """Eyes cut hard to one side while the head stays put."""
    return abs(f.gaze) > THRESH["side_eye"] and f.blink < 0.5


def _horse(f: Face) -> bool:
    """Wide-eyed stare, mouth shut, nothing in your hands."""
    return f.eye_wide > THRESH["stare_wide"] and f.jaw_open < THRESH["stare_jaw"]


# Order matters: the most constrained rule goes first, so a looser one cannot
# steal a pose that belongs to it.
TWO_HAND_RULES = (("HEART", _heart), ("HANDS_MOUTH", _hands_mouth),
                  ("TIMEOUT", _timeout), ("DOUBLE_THUMBS", _double_thumbs),
                  ("MIND_BLOWN", _mind_blown))
FACE_HAND_RULES = (("SCUBA", _scuba),)
FREE_HAND_RULES = (("CALL_ME", _call_me), ("THUMBS_UP", _thumbs_up))


def _hands_raised(hands, f: Face) -> bool:
    """Both hands up around the head - i.e. a MIND_BLOWN in progress."""
    top = f.y + THRESH["raised_hands"] * f.h
    return len(hands) >= 2 and sum(h.palm[1] < top for h in hands) >= 2


def classify(hands, face):
    """Return a gesture name or None.  Order matters: the most specific poses are
    tested first so a looser rule can't steal them."""
    if face is None:
        return None
    fw = max(face.w, 1.0)

    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        for name, rule in TWO_HAND_RULES:
            if rule(a, b, face, fw):
                return name

    for h in hands:                       # hand-on-face poses
        for name, rule in FACE_HAND_RULES:
            if rule(h, face, fw):
                return name

    # With both hands up by the head, a single-hand pose is almost always a misread
    # of a MIND_BLOWN mid-flight - one rising hand reading as thumb-only was firing
    # THUMBS_UP over the top of it. Better to show nothing for a frame.
    if not _hands_raised(hands, face):
        for h in hands:                   # free-standing poses
            for name, rule in FREE_HAND_RULES:
                if rule(h, face, fw):
                    return name

    if _tongue(face):                     # expressions come last
        return "TONGUE"
    if _wink(face):
        return "WINK"
    if _side_eye(face):
        return "SIDE_EYE"
    if not hands and _horse(face):
        return "HORSE"
    return None


class Stabilizer:
    """Debounce: a gesture must be held for `on_frames` before it fires, and must
    be absent for `off_frames` (and held a minimum time) before it releases.
    This is what stops the overlay strobing on borderline frames."""

    def __init__(self, on_frames=3, off_frames=8, min_hold_s=0.7, per_gesture=ON_FRAMES):
        self.on_frames, self.off_frames, self.min_hold_s = on_frames, off_frames, min_hold_s
        self.per_gesture = per_gesture or {}
        self.active = None
        self.started_at = 0.0
        self._cand, self._cand_n, self._miss = None, 0, 0

    def _needed(self, gesture) -> int:
        return max(self.on_frames, self.per_gesture.get(gesture, 0))

    def update(self, raw, now=None):
        now = time.monotonic() if now is None else now

        if self.active is not None:
            if raw == self.active:
                self._miss = 0
            else:
                self._miss += 1
                if self._miss >= self.off_frames and now - self.started_at >= self.min_hold_s:
                    self.active, self._miss = None, 0
                    self._cand, self._cand_n = raw, 1 if raw else 0
            return self.active

        if raw is None:
            self._cand, self._cand_n = None, 0
            return None

        self._cand_n = self._cand_n + 1 if raw == self._cand else 1
        self._cand = raw
        if self._cand_n >= self._needed(raw):
            self.active, self.started_at, self._miss = raw, now, 0
        return self.active


# --------------------------------------------------------------------------- #
# Several people in one frame
# --------------------------------------------------------------------------- #
def assign_hands(hands, faces, max_reach=2.2, per_face=2):
    """Hand out each hand to the face it most plausibly belongs to.

    Distance is measured in face widths, so the person at the back of the room is
    judged on the same scale as the one leaning into the camera. Greedy over all
    (hand, face) pairs sorted by distance: a hand whose nearest face already has
    two hands falls through to the next-best face rather than being dropped.
    """
    out = [[] for _ in faces]
    if not faces or not hands:
        return out
    pairs = sorted((_d(h.palm, f.center) / max(f.w, 1.0), hi, fi)
                   for hi, h in enumerate(hands) for fi, f in enumerate(faces))
    claimed = set()
    for dist, hi, fi in pairs:
        if dist > max_reach or hi in claimed or len(out[fi]) >= per_face:
            continue
        out[fi].append(hands[hi])
        claimed.add(hi)
    return out


class Person:
    """One tracked person: smoothed face, their own gesture debouncer."""

    __slots__ = ("id", "tracker", "stab", "face", "hands", "last_seen")

    def __init__(self, person_id: int, stabilizer: "Stabilizer", alpha=0.45, coast_s=0.5):
        self.id = person_id
        self.tracker = FaceTracker(alpha=alpha, coast_s=coast_s)
        self.stab = stabilizer
        self.face = None
        self.hands = []
        self.last_seen = 0.0

    def update(self, vec, now):
        self.face = self.tracker.update(vec, 1.0 if vec is not None else 0.0, now)
        if vec is not None:
            self.last_seen = now
        return self.face


class PersonTracker:
    """Keeps identities stable across frames so each person keeps their own meme.

    MediaPipe hands back an unordered list of faces with no ids, and the order can
    change between frames; without matching, two people would swap overlays (and
    each other's trigger state) whenever the detector reordered them.
    """

    def __init__(self, stabilizer_factory, max_people=5, match_dist=0.9,
                 alpha=0.45, coast_s=0.5):
        self.new_stabilizer = stabilizer_factory
        self.max_people = max_people
        self.match_dist = match_dist
        self.alpha, self.coast_s = alpha, coast_s
        self._people = []
        self._next_id = 1

    def update(self, face_vecs, now):
        vecs = list(face_vecs or [])
        pairs = []
        for vi, vec in enumerate(vecs):
            centre = np.array([vec[0] + vec[2] / 2.0, vec[1] + vec[3] / 2.0])
            for p in self._people:
                if p.face is None:
                    continue
                pairs.append((float(np.linalg.norm(centre - p.face.center))
                              / max(float(vec[2]), 1.0), vi, p))
        pairs.sort(key=lambda t: t[0])

        matched_v, matched_p = set(), set()
        for dist, vi, p in pairs:
            if dist > self.match_dist or vi in matched_v or id(p) in matched_p:
                continue
            matched_v.add(vi)
            matched_p.add(id(p))
            p.update(vecs[vi], now)

        for vi, vec in enumerate(vecs):                     # newcomers
            if vi in matched_v or len(self._people) >= self.max_people:
                continue
            person = Person(self._next_id, self.new_stabilizer(),
                            alpha=self.alpha, coast_s=self.coast_s)
            self._next_id += 1
            person.update(vec, now)
            self._people.append(person)

        for p in self._people:                              # coast the unmatched
            if id(p) not in matched_p:
                p.update(None, now)

        self._people = [p for p in self._people if p.face is not None]
        return self._people
