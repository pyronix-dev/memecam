#!/usr/bin/env python3
"""MemeCam - gesture-triggered meme overlays, piped into a virtual camera.

    python memecam.py                 # webcam -> overlays -> OBS Virtual Camera
    python memecam.py --preview       # + local preview window (hotkeys below)
    python memecam.py --no-vcam --preview --debug   # tuning mode, no virtual cam

Hotkeys in the preview window:
    q / ESC  quit          d  toggle debug HUD      m  toggle mirror
    1..8     force that gesture for 2s (check alignment without contorting)
"""
from __future__ import annotations

import argparse
import collections
import sys
import threading
import time

import cv2
import numpy as np

from gesture_rules import (ON_FRAMES, THRESH, Face, PersonTracker, Stabilizer,
                           assign_hands, classify)
from meme_assets import GESTURE_HOWTO, GESTURE_STYLE, load_all
from overlay_fx import alpha_blend, rotate_rgba

# Keys that force an overlay for 2 s, in GESTURE_STYLE order.
FORCE_CHARS = "123456789zxcvb"
FORCE_KEYS = dict(zip(FORCE_CHARS, GESTURE_STYLE.keys()))
FORCE_HELP = "".join(FORCE_KEYS)
MAX_OVERLAY_FRAC = 0.9      # one overlay may never eat more than this much of the frame


# --------------------------------------------------------------------------- #
# Overlay placement
# --------------------------------------------------------------------------- #
def draw_overlay(frame, sprite, elapsed_s, face: Face, style) -> None:
    """Scale / rotate / position one sprite against the detected face."""
    anchor = style["anchor"]

    if anchor == "eyes":
        # Width is driven by the eye distance and the sprite is rotated onto the
        # eye line, so shades and heart-eyes stay glued on when you tilt your head.
        target_w = int(style["scale"] * face.eye_dist)
        img = sprite.get(elapsed_s, target_w)
        dx, dy = face.left_eye - face.right_eye
        img = rotate_rgba(img, -np.degrees(np.arctan2(dy, dx)))
        cx, cy = (face.left_eye + face.right_eye) / 2.0
        cy += style.get("dy", 0.0) * face.eye_dist
        cx += style.get("dx", 0.0) * face.eye_dist
    else:
        # Cover-fit: scale so the meme covers the face box on BOTH axes, taking the
        # GIF's own aspect ratio into account. Sizing on width alone made a tall
        # portrait GIF tower over a face and a wide one sit across it like a letterbox
        # - and the mismatch got worse the further someone sat from the camera.
        sh, sw = sprite.frames[0].shape[:2]
        if anchor == "side":
            cover = style["scale"] * face.w        # parked beside you, not covering
        else:
            cover = style["scale"] * max(face.w, face.h * sw / float(sh))
        target_w = int(min(cover, frame.shape[1] * MAX_OVERLAY_FRAC))
        img = sprite.get(elapsed_s, target_w)
        cx = face.cx + style.get("dx", 0.0) * face.w
        cy = face.cy + style.get("dy", 0.0) * face.h
        if anchor == "above":                      # sit on top of the head
            cy = face.y - img.shape[0] / 2.0 + style.get("dy", 0.0) * face.h
            cy = max(cy, img.shape[0] / 2.0 + 4)    # keep it on-screen if you sit high
        elif anchor == "side":
            # Park it beside your head, on whichever side has more room, so the
            # overlay reacts *next to* you instead of hiding your face.
            half = img.shape[1] / 2.0
            gap = 0.12 * face.w
            room_right = frame.shape[1] - (face.x + face.w)
            if room_right >= face.x:
                cx = face.x + face.w + gap + half
            else:
                cx = face.x - gap - half
            cx = min(max(cx, half + 4), frame.shape[1] - half - 4)

    h, w = img.shape[:2]
    alpha_blend(frame, img, int(round(cx - w / 2.0)), int(round(cy - h / 2.0)))


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
def _text(frame, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2,
                cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame, fps, tracked, hands, missing, trigger_ms=0.0, dropped=0):
    _text(frame, f"{fps:5.1f} fps   people:{len(tracked)}   hands:{len(hands)}"
                 f"   trigger:{trigger_ms:4.0f}ms   stale dropped:{dropped}",
          (12, 24), 0.55)
    if missing:
        _text(frame, "missing assets: " + ", ".join(sorted(missing)),
              (12, frame.shape[0] - 14), 0.45, (80, 160, 255))
    for row, person in enumerate(tracked):
        f = person.face
        cv2.rectangle(frame, (int(f.x), int(f.y)),
                      (int(f.x + f.w), int(f.y + f.h)), (0, 200, 255), 1)
        _text(frame, f"#{person.id} {person.stab.active or '-'}",
              (int(f.x), max(int(f.y) - 8, 14)), 0.5, (0, 200, 255))
        bits = " ".join(f"{h.label[:1]}:{h.fingers_str()}" for h in person.hands)
        _text(frame, f"#{person.id} jaw:{f.jaw_open:.2f} wide:{f.eye_wide:.2f} "
                     f"tongue:{f.tongue:.2f} wink:{abs(f.blink_l - f.blink_r):.2f} "
                     f"gaze:{abs(f.gaze):.2f}  {bits}",
              (12, 48 + 20 * row), 0.48, (120, 200, 255))
        for h in person.hands:
            cv2.circle(frame, tuple(int(v) for v in h.palm), 5, (120, 200, 255), -1)


# --------------------------------------------------------------------------- #
# Camera helpers
# --------------------------------------------------------------------------- #
class DetectWorker:
    """Runs one detection stage on its own thread.

    Measured at 1280x720 on this machine: hands cost ~29 ms/frame, and faces cost
    ~3.8 ms for the detector plus ~6.7 ms per person - about 37 ms for five people.
    Run sequentially in the capture loop that is 66 ms, i.e. 15 fps, so the video
    would stutter and gestures would land late. Given a thread each, the main loop
    only does capture and compositing and holds camera rate; results are at most one
    detection old, which is invisible for a pose you hold for half a second.
    """

    def __init__(self, fn, name="worker"):
        self.fn, self.name = fn, name
        self._cond = threading.Condition()
        self._pending = None
        self._result = []
        self.processed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, rgb, ts_ms):
        """Hand over the newest frame, discarding any frame not started yet."""
        with self._cond:
            self._pending = (rgb, ts_ms)
            self._cond.notify()

    def latest(self):
        with self._cond:
            return self._result

    def _run(self):
        while not self._stop.is_set():
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(timeout=0.2)
                if self._stop.is_set():
                    return
                rgb, ts = self._pending
                self._pending = None
            try:
                result = self.fn(rgb, ts)
            except Exception:
                result = []
            with self._cond:
                self._result = result
                self.processed += 1

    def close(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)


class CameraStream:
    """Background grabber that always hands back the NEWEST frame.

    A webcam keeps filling its queue while you work; reading it in the main loop
    means that whenever a frame costs a little too long you start serving video
    from the past, and the lag never recovers. This thread drains the camera and
    keeps only the latest frame, so the gesture you just made is the one on screen.
    """

    def __init__(self, cap):
        self.cap = cap
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._served = -1
        self.dropped = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception:
                break
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._cond:
                if self._frame is not None and self._seq != self._served:
                    self.dropped += 1          # main loop never saw the last one
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self, timeout=2.0):
        """Block until a frame we have not served yet is ready.

        A condition variable, not a polling sleep: polling at 1 ms granularity cost
        ~9 ms per frame here, which is a third of a 30 fps frame budget.
        """
        with self._cond:
            if self._cond.wait_for(
                    lambda: self._frame is not None and self._seq != self._served,
                    timeout=timeout):
                self._served = self._seq
                return True, self._frame
        return False, None

    def get(self, prop):
        return self.cap.get(prop)

    def set(self, prop, value):
        return self.cap.set(prop, value)

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


def open_camera(index, width, height, fps):
    backends = ([cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == "darwin"
                else [cv2.CAP_ANY])
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # keep latency down
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame
            cap.release()
    return None, None


def open_vcam(width, height, fps):
    try:
        import pyvirtualcam
    except ImportError:
        print("! pyvirtualcam is not installed - running preview-only.\n"
              "  pip install pyvirtualcam")
        return None, None
    try:
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                  fmt=pyvirtualcam.PixelFormat.RGB)
        print(f"* Virtual camera: {cam.device}  ({width}x{height} @ {fps}fps)")
        return cam, pyvirtualcam
    except Exception as exc:
        print(f"! Could not open a virtual camera: {exc}\n"
              "  macOS: install OBS (>=26.1) and click 'Start Virtual Camera' once so\n"
              "  the system camera extension is registered, then re-run.\n"
              "  Falling back to preview-only.")
        return None, None


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Gesture-triggered meme overlays -> virtual cam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preview", action="store_true", help="show a local preview window")
    ap.add_argument("--no-vcam", action="store_true", help="skip the virtual camera")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the image")
    ap.add_argument("--debug", action="store_true", help="start with the HUD on")
    ap.add_argument("--hold", type=float, default=0.7, help="min seconds an overlay stays up")
    ap.add_argument("--on-frames", type=int, default=2, help="frames to trigger a gesture")
    ap.add_argument("--off-frames", type=int, default=8, help="frames to release a gesture")
    ap.add_argument("--det-conf", type=float, default=0.5,
                    help="MediaPipe detection confidence; lower = hands picked up sooner")
    ap.add_argument("--max-people", type=int, default=5, metavar="N",
                    help="how many faces to track and overlay independently (1-5)")
    ap.add_argument("--no-threads", action="store_true",
                    help="run detection inline instead of on worker threads "
                         "(simpler, but roughly halves the frame rate)")
    ap.add_argument("--fast", action="store_true",
                    help="snappiest triggering: fire on the first matching frame, "
                         "shorter hold, lower detection confidence")
    ap.add_argument("--thresh", action="append", default=[], metavar="KEY=VALUE",
                    help="override a gesture_rules.THRESH entry, e.g. --thresh tongue=0.18 "
                         "(repeatable; --thresh list prints them all)")
    ap.add_argument("--list-gestures", action="store_true")
    args = ap.parse_args()

    per_gesture = dict(ON_FRAMES)
    if args.fast:
        args.on_frames = 1
        args.off_frames = min(args.off_frames, 6)
        args.hold = min(args.hold, 0.35)
        args.det_conf = min(args.det_conf, 0.45)
        per_gesture = {g: max(1, n // 2) for g, n in per_gesture.items()}

    for item in args.thresh:
        if item == "list":
            for k, v in THRESH.items():
                print(f"  {k:<18} {v}")
            return 0
        key, _, value = item.partition("=")
        if key not in THRESH:
            print(f"! unknown threshold {key!r}; --thresh list shows them all")
            return 1
        THRESH[key] = float(value)
        print(f"* threshold {key} = {THRESH[key]}")

    if args.list_gestures:
        for ch, g in FORCE_KEYS.items():
            print(f"  [{ch}] {g:<12} {GESTURE_HOWTO[g]:<50} -> "
                  f"{GESTURE_STYLE[g]['file']}")
        return 0

    from detector import Detector              # imported late: MediaPipe is slow to load
    from fetch_models import ensure_models

    print("Loading assets...")
    sprites = load_all()
    missing = set(GESTURE_STYLE) - set(sprites)
    if not sprites:
        print("! No assets found. Run:  python fetch_gifs.py")
        return 1

    cap, first = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is not None:
        cap = CameraStream(cap)
    if cap is None:
        print(f"! Could not open camera {args.camera}.\n"
              "  macOS: the first run pops a Camera permission dialog - say Allow.\n"
              "  If it was denied before, enable your terminal under\n"
              "  System Settings > Privacy & Security > Camera (or run 'tccutil reset Camera'\n"
              "  in this terminal to get the prompt back), then re-run.")
        return 1
    H, W = first.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 0) or args.fps
    print(f"* Webcam: {W}x{H} @ {fps}fps")

    cam, _ = (None, None) if args.no_vcam else open_vcam(W, H, fps)
    show_preview = args.preview or cam is None
    if cam is None and not args.no_vcam:
        show_preview = True

    try:
        detector = Detector(det_conf=args.det_conf, max_people=args.max_people)
    except FileNotFoundError:
        print("* Fetching MediaPipe models (one time, ~8 MB)...")
        ensure_models()
        detector = Detector(det_conf=args.det_conf, max_people=args.max_people)

    def new_stabilizer():
        return Stabilizer(on_frames=args.on_frames, off_frames=args.off_frames,
                          min_hold_s=args.hold, per_gesture=per_gesture)

    hand_worker = face_worker = None
    if not args.no_threads:
        hand_worker = DetectWorker(
            lambda rgb, ts: detector.process_hands(rgb, ts, out_size=(W, H)), "hands")
        face_worker = DetectWorker(detector.process_faces, "faces")
    people = PersonTracker(new_stabilizer, max_people=args.max_people)
    show_hud = args.debug
    last_raw, raw_since, prev_active, trigger_ms = None, 0.0, None, 0.0
    mirror = not args.no_mirror
    forced_until, forced_gesture = 0.0, None
    frame_times = collections.deque(maxlen=30)

    print("\nGestures:")
    for ch, g in FORCE_KEYS.items():
        mark = " " if g in sprites else "!"
        print(f"  {mark} [{ch}] {g:<12} {GESTURE_HOWTO[g]}")
    print("\nRunning. Ctrl-C (or 'q' in the preview window) to stop.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("! No frames from the webcam for 2 s - stopping.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()

            # --- detection (MediaPipe wants contiguous RGB)
            # Detection runs on the full-resolution frame on purpose: feeding
            # MediaPipe a downscaled copy shifted the face box by ~40 px and made
            # the inner-lip landmarks unusable for the tongue test.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int(now * 1000)
            if hand_worker is not None:
                hand_worker.submit(rgb, ts)
                face_worker.submit(rgb, ts)
                hands, face_vecs = hand_worker.latest(), face_worker.latest()
            else:
                hands, face_vecs = detector.process(rgb, ts, out_size=(W, H))

            # --- one gesture, one overlay, one debouncer per person
            tracked = people.update(face_vecs, now)
            per_person = assign_hands(hands, [p.face for p in tracked])
            fired = []
            for person, own_hands in zip(tracked, per_person):
                person.hands = own_hands
                raw = classify(own_hands, person.face)
                active = person.stab.update(raw, now)
                if forced_gesture and now < forced_until:
                    active = forced_gesture
                    elapsed = now - (forced_until - 2.0)
                else:
                    elapsed = now - person.stab.started_at
                if active and active in sprites:
                    draw_overlay(frame, sprites[active], elapsed, person.face,
                                 GESTURE_STYLE[active])
                    fired.append((person.id, active))

            if forced_gesture and now >= forced_until:
                forced_gesture = None

            # trigger-latency read-out follows whichever person fired first
            raw_now = fired[0][1] if fired else None
            if raw_now is not None and raw_now != last_raw:
                raw_since = now
            last_raw = raw_now
            if raw_now is not None and raw_now != prev_active:
                trigger_ms = (now - raw_since) * 1000.0
            prev_active = raw_now

            frame_times.append(now)
            if show_hud:
                span = frame_times[-1] - frame_times[0]
                cur_fps = (len(frame_times) - 1) / span if span > 0 else 0.0
                draw_hud(frame, cur_fps, tracked, hands, missing,
                         trigger_ms, cap.dropped)

            if cam is not None:
                cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()

            if show_preview:
                cv2.imshow(f"MemeCam (q=quit  d=debug  m=mirror  {FORCE_HELP}=force)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("d"):
                    show_hud = not show_hud
                if key == ord("m"):
                    mirror = not mirror
                forced = FORCE_KEYS.get(chr(key) if 32 <= key < 127 else "")
                if forced:
                    forced_gesture, forced_until = forced, now + 2.0
    except KeyboardInterrupt:
        pass
    finally:
        for w in (hand_worker, face_worker):
            if w is not None:
                w.close()
        cap.release()
        detector.close()
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
