#!/usr/bin/env python3
"""Synthetic-landmark regression test for the gesture rules.

Builds a fake hand skeleton for each of the 8 gestures and asserts that classify()
returns it - and, just as important, that no other pose steals it.

    python test_gestures.py
"""
from __future__ import annotations

import sys

import numpy as np

from gesture_rules import (Face, PersonTracker, Stabilizer, _thumbs_up,
                           assign_hands, classify)

FW, FH = 240.0, 300.0                    # synthetic face box
CX, CY = 640.0, 340.0
FX, FY = CX - FW / 2, CY - FH / 2
NOSE = np.array([CX, CY + 0.06 * FH])
MOUTH = np.array([CX, CY + 0.24 * FH])
SPAN = 0.55 * FW                         # wrist -> middle-knuckle length


def face_vec(jaw=0.05, wide=0.02, tongue=0.0, dx=0.0,
             blink_l=0.10, blink_r=0.10, gaze=0.0) -> np.ndarray:
    return np.array([
        FX + dx, FY, FW, FH,
        CX + dx - 0.21 * FW, CY - 0.12 * FH,   # right eye
        CX + dx + 0.21 * FW, CY - 0.12 * FH,   # left eye
        NOSE[0] + dx, NOSE[1], MOUTH[0] + dx, MOUTH[1],
        jaw, wide, blink_l, blink_r, gaze, tongue], dtype=np.float32)


def face(jaw=0.05, wide=0.02, tongue=0.0, dx=0.0,
         blink_l=0.10, blink_r=0.10, gaze=0.0) -> Face:
    return Face(face_vec(jaw, wide, tongue, dx, blink_l, blink_r, gaze))


class FakeLm:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x, self.y = x, y


def hand(palm, direction=(0, -1), ext=(1, 1, 1, 1, 1), span=SPAN, thumb_side=1.0):
    """Build 21 landmarks for a stylised hand whose palm centre lands on `palm`.

    ext = (thumb, index, middle, ring, pinky); 1 = extended, 0 = curled.
    """
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    perp = np.array([-d[1], d[0]]) * thumb_side          # across the knuckles
    wrist = np.zeros(2)
    pts = [wrist]

    # thumb chain (1-4), swung out to the side away from the pinky
    tdir = (d * 0.55 + perp * 0.85)
    tdir /= np.linalg.norm(tdir)
    reach = (0.30, 0.55, 0.80, 1.05) if ext[0] else (0.22, 0.34, 0.40, 0.42)
    for r in reach:
        pts.append(wrist + tdir * span * r)

    # four fingers (5-20): MCP row across the knuckles, then PIP/DIP/TIP up `d`
    for i, offset in enumerate((0.34, 0.10, -0.14, -0.38)):
        mcp = wrist + d * span * 0.92 + perp * span * offset
        if ext[i + 1]:
            chain = (0.34, 0.62, 0.88)
        else:
            chain = (0.30, 0.16, 0.02)                   # curled back toward the palm
        pts.append(mcp)
        for r in chain:
            pts.append(mcp + d * span * r)

    pts = np.array(pts)
    pts += np.asarray(palm, dtype=np.float64) - pts[[0, 5, 9, 13, 17]].mean(axis=0)
    return [FakeLm(p[0], p[1]) for p in pts]


def H(landmarks, label="Right"):
    from gesture_rules import Hand
    return Hand(landmarks, 1, 1, label)                  # already in pixels


def move_point(lms, idx, target):
    lms[idx] = FakeLm(float(target[0]), float(target[1]))
    return lms


# --------------------------------------------------------------------------- #
def poses():
    """name -> (hands, face) for every gesture."""
    out = {}

    out["HANDS_MOUTH"] = ([H(hand((CX - 0.18 * FW, MOUTH[1] + 0.05 * FH), (0, -1))),
                           H(hand((CX + 0.18 * FW, MOUTH[1] + 0.05 * FH), (0, -1),
                                  thumb_side=-1))], face())

    out["MIND_BLOWN"] = ([H(hand((CX - 0.85 * FW, FY - 0.05 * FH), (0, -1))),
                          H(hand((CX + 0.85 * FW, FY - 0.05 * FH), (0, -1),
                                 thumb_side=-1))], face())

    # heart: index tips meet high, thumb tips meet lower
    a = hand((CX - 0.30 * FW, CY + 0.75 * FH), (0.6, -1), ext=(1, 1, 0, 0, 0))
    b = hand((CX + 0.30 * FW, CY + 0.75 * FH), (-0.6, -1), ext=(1, 1, 0, 0, 0),
             thumb_side=-1)
    apex = np.array([CX, CY + 0.52 * FH])
    base = np.array([CX, CY + 0.80 * FH])
    move_point(a, 8, apex + [-2, 0]); move_point(b, 8, apex + [2, 0])
    move_point(a, 4, base + [-2, 0]); move_point(b, 4, base + [2, 0])
    out["HEART"] = ([H(a), H(b)], face())

    # time-out T: flat horizontal hand, vertical hand's middle tip under its palm
    horiz = hand((CX, CY + 0.70 * FH), (1, 0))
    vert = hand((CX, CY + 1.05 * FH), (0, -1))
    move_point(vert, 12, (CX, CY + 0.74 * FH))
    out["TIMEOUT"] = ([H(vert), H(horiz)], face())

    pinch = hand((CX + 0.35 * FW, NOSE[1] + 0.25 * FH), (0, -1), ext=(1, 1, 0, 0, 0))
    move_point(pinch, 8, NOSE + [3, 2]); move_point(pinch, 4, NOSE + [-3, 2])
    out["SCUBA"] = ([H(pinch)], face())

    up = hand((CX + 1.25 * FW, CY), (1, 0), ext=(1, 0, 0, 0, 0), thumb_side=-1)
    out["THUMBS_UP"] = ([H(up)], face())

    out["DOUBLE_THUMBS"] = ([H(hand((CX - 1.25 * FW, CY), (-1, 0), ext=(1, 0, 0, 0, 0))),
                             H(hand((CX + 1.25 * FW, CY), (1, 0), ext=(1, 0, 0, 0, 0),
                                    thumb_side=-1))], face())

    out["CALL_ME"] = ([H(hand((CX + 1.20 * FW, CY), (0, -1), ext=(1, 0, 0, 0, 1)))], face())

    out["TONGUE"] = ([], face(jaw=0.45, tongue=0.6))
    out["WINK"] = ([], face(blink_l=0.90, blink_r=0.05))
    out["SIDE_EYE"] = ([], face(gaze=0.70))
    out["HORSE"] = ([], face(wide=0.5))
    out[None] = ([], face())                              # resting: no overlay
    return out


def shift(hand_landmarks, dx):
    return [FakeLm(p.x + dx, p.y) for p in hand_landmarks]


def collision_checks():
    """Poses that used to be misread as something else."""
    out = []
    top = FY - 0.05 * FH

    # Mid-flight MIND_BLOWN: both hands are up, but one momentarily reads as
    # thumb-only with the thumb pointing up. On its own that hand IS a valid
    # thumbs up - which is exactly how it used to fire over the top of the gesture.
    rogue = H(hand((CX + 0.85 * FW, top), (1, 0), ext=(1, 0, 0, 0, 0), thumb_side=-1))
    out.append(("the misread hand really does satisfy the thumbs-up rule",
                _thumbs_up(rogue, face(), FW)))
    misread = [H(hand((CX - 0.85 * FW, top), (0, -1))), rogue]
    got = classify(misread, face())
    out.append((f"...but with both hands raised it is suppressed -> {got}",
                got != "THUMBS_UP"))

    # Two thumbs up held high are a gesture in their own right, not suppressed.
    both_thumbs = [H(hand((CX - 0.85 * FW, top), (-1, 0), ext=(1, 0, 0, 0, 0))),
                   H(hand((CX + 0.85 * FW, top), (1, 0), ext=(1, 0, 0, 0, 0),
                          thumb_side=-1))]
    out.append(("both thumbs up, raised high -> DOUBLE_THUMBS",
                classify(both_thumbs, face()) == "DOUBLE_THUMBS"))

    # Both hands genuinely open and up still reads as MIND_BLOWN.
    both_open = [H(hand((CX - 0.85 * FW, top), (0, -1))),
                 H(hand((CX + 0.85 * FW, top), (0, -1), thumb_side=-1))]
    out.append(("both hands open and raised -> MIND_BLOWN",
                classify(both_open, face()) == "MIND_BLOWN"))

    # A single thumbs up, hands down, still works.
    one = poses()["THUMBS_UP"][0]
    out.append(("a lone thumbs up still fires",
                classify(one, face()) == "THUMBS_UP"))

    # An ordinary blink shuts both eyes and must not read as a wink.
    out.append(("a normal blink is not a WINK",
                classify([], face(blink_l=0.85, blink_r=0.82)) != "WINK"))
    return out


def crowd_checks():
    """Two people in one frame: each must get their own hands and their own meme."""
    out = []
    gap = 3.2 * FW                                  # person B, well clear of person A

    a_hands, _ = poses()["SCUBA"]                   # A pinches their nose
    b_pose, _ = poses()["THUMBS_UP"]                # B gives a thumbs up
    b_hands = [H(shift([FakeLm(p[0], p[1]) for p in h.pts], gap)) for h in b_pose]

    faces = [face(), face(dx=gap)]
    hands = list(a_hands) + list(b_hands)
    owned = assign_hands(hands, faces)
    out.append(("hands split 1/1 between the two people",
                [len(o) for o in owned] == [1, 1]))
    out.append(("person A -> SCUBA", classify(owned[0], faces[0]) == "SCUBA"))
    out.append(("person B -> THUMBS_UP", classify(owned[1], faces[1]) == "THUMBS_UP"))
    out.append(("A's hand is not given to B",
                owned[1] and abs(owned[1][0].palm[0] - faces[1].cx) < 2.2 * FW))

    # identities must survive the detector returning the faces in a different order
    tracker = PersonTracker(lambda: Stabilizer(), max_people=5)
    vecs = [face_vec(), face_vec(dx=gap)]
    first = {p.id: round(float(p.face.cx)) for p in tracker.update(vecs, 0.0)}
    second = {p.id: round(float(p.face.cx))
              for p in tracker.update(list(reversed(vecs)), 0.05)}
    out.append(("ids stable when the face list is reordered",
                len(first) == 2 and first == second))
    return out


def main() -> int:
    results, failures = [], 0
    for expected, (hands, f) in poses().items():
        got = classify(hands, f)
        ok = got == expected
        failures += not ok
        results.append(f"  {'PASS' if ok else 'FAIL'}  expected {str(expected):<12} got {got}")
    for label, ok in collision_checks() + crowd_checks():
        failures += not ok
        results.append(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print("\n".join(results))
    print(f"\n{len(results) - failures}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
