<div align="center">

# 🎭 MemeCam

### Put memes on your face in video calls, using hand gestures.

[![Python](https://img.shields.io/badge/Python-3.10%20–%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-00A6A6?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![macOS](https://img.shields.io/badge/macOS-Apple_Silicon-000000?style=for-the-badge&logo=apple&logoColor=white)](#-notes)

![Gestures](https://img.shields.io/badge/gestures-12-FF6B6B?style=flat-square)
![People](https://img.shields.io/badge/people_at_once-5-4ECDC4?style=flat-square)
![Speed](https://img.shields.io/badge/58_fps-with_5_people-51CF66?style=flat-square)
![Tests](https://img.shields.io/badge/tests-24%2F24_passing-51CF66?style=flat-square)
![Privacy](https://img.shields.io/badge/runs-100%25_locally-845EF7?style=flat-square)

</div>

<!-- Drop a demo GIF here once you have one:  ![demo](docs/demo.gif)  -->

MemeCam reads your webcam, watches for a dozen gestures and expressions with **MediaPipe**,
drops the matching meme GIF over your face, and publishes the result as a **virtual
camera** — so you can pick it in Zoom, Google Meet or Teams like any other webcam.

🤿 Pinch your nose and you're a scuba cat. 🤯 Throw your hands up and your head explodes.
😐 Do nothing and your normal webcam feed passes straight through, untouched.

Up to **five people** in the same frame each get their own gesture, their own meme and
their own timing.

---

## 🚀 Quick start

```bash
git clone https://github.com/pyronix-dev/memecam.git
cd memecam
./setup.sh          # virtualenv + dependencies + models + GIFs
./run.sh --preview
```

Then in Zoom (**Settings → Video → Camera**) or the browser camera dropdown for
Meet/Teams, pick **OBS Virtual Camera**.

> [!IMPORTANT]
> **Two things the setup needs.**
> **Python 3.10–3.12** — `brew install python@3.12` on macOS. The system Python 3.9 is too
> old for MediaPipe and 3.13 has no wheels yet; `setup.sh` checks and tells you.
> **[OBS](https://obsproject.com) installed, and *Start Virtual Camera* clicked once**, so
> macOS registers the camera device. After that OBS never needs to run again — MemeCam
> writes to the device directly.

> [!TIP]
> No interest in the virtual camera? `./run.sh --no-vcam --preview` just opens a window.

---

## 🎭 The gestures

Hold one and the meme appears; drop it and your normal feed comes back.

| key | gesture | how to trigger it | meme |
|:---:|:---|:---|:---|
| `1` | ❤️ `HEART` | two-hand heart — index tips and thumb tips touching | Shaq heart |
| `2` | ✋ `TIMEOUT` | time-out "T": one hand flat, the other vertical under it | wait… hold on |
| `3` | 🤿 `SCUBA` | pinch your nose shut (thumb + index on the nose) | scuba cat |
| `4` | 🤯 `MIND_BLOWN` | both open hands up beside your head, fingers spread | brain explode |
| `5` | 😛 `TONGUE` | stick your tongue out | dog tongue |
| `6` | 🐴 `HORSE` | wide-eyed stare, no hands in shot, hold it | horse stare |
| `7` | 🙊 `HANDS_MOUTH` | both hands clapped over your **mouth** | Higuruma (JJK) |
| `8` | 👍 `THUMBS_UP` | **one** thumbs up, away from your face → lands *beside* your head | thumbs up |
| `9` | 👍👍 `DOUBLE_THUMBS` | **both** thumbs up | Shaun the Sheep |
| `z` | 😉 `WINK` | one eye shut, the other open — hold it | wink |
| `x` | 🤙 `CALL_ME` | thumb + pinky out, middle fingers folded | DJ Khaled "call me" |
| `c` | 👀 `SIDE_EYE` | cut your eyes to one side without turning your head | side-eye monkey |

> [!TIP]
> In the preview window those keys **force** the matching overlay for two seconds — the
> quickest way to see how each one looks without contorting yourself.
> Other keys: <kbd>q</kbd>/<kbd>Esc</kbd> quit · <kbd>d</kbd> debug HUD · <kbd>m</kbd> mirror.

The overlay is the image alone — no caption, no label bar.

---

## 🎬 Running it

```bash
./run.sh --preview                    # virtual camera + a preview window
./run.sh                              # virtual camera only, no window
./run.sh --fast --preview             # fire on the first matching frame
./run.sh --max-people 2               # cap how many faces get overlays (default 5)
./run.sh --no-vcam --preview --debug  # tuning mode: HUD on, no virtual camera
./run.sh --list-gestures
./run.sh --thresh list                # print every tunable threshold
```

Also useful: `--camera N` · `--width/--height/--fps` · `--no-mirror` ·
`--on-frames`/`--off-frames`/`--hold` (trigger timing) · `--no-threads` (run detection
inline — simpler, much slower).

> [!NOTE]
> macOS asks for camera permission on the first run. If you denied it once, re-enable your
> terminal under **System Settings → Privacy & Security → Camera**, or run
> `tccutil reset Camera` in that terminal to get the prompt back.

---

## 🎨 Swapping in your own memes

Every meme is a Giphy link in `GIPHY_LINKS` at the top of `fetch_gifs.py`. To repoint a
gesture, paste a new Giphy page URL:

```bash
./.venv/bin/python fetch_gifs.py SCUBA https://giphy.com/gifs/some-other-meme-xxxxx
```

It takes a page URL, a media URL or a bare GIF id and saves `assets/<file>.gif`, where
`<file>` comes from `GESTURE_STYLE` in `meme_assets.py`. Edit `GIPHY_LINKS` to make it
permanent. The loader prefers `.gif` but also reads `.webp`, `.apng` and `.png` with real
transparency, so your own artwork works too.

How each meme is placed is one line in `meme_assets.py`:

```python
"SCUBA": dict(file="scuba", anchor="face", scale=1.35, dx=0.0, dy=-0.05),
```

| field | what it does |
|:---|:---|
| `anchor` | `face` centred on your face · `side` beside your head, face still visible · `above` on top of your head · `eyes` rotated onto your eye line, for glasses-style art |
| `scale` | margin factor — `1.0` is exactly your face box, `1.35` leaves ~35% around it |
| `dx` / `dy` | nudge it, in face-box units |

**Adding a whole new gesture:** write a rule in `gesture_rules.py`, register it in
`TWO_HAND_RULES` / `FACE_HAND_RULES` / `FREE_HAND_RULES`, add entries to `GESTURE_STYLE`
and `GESTURE_HOWTO`, add its Giphy link to `GIPHY_LINKS`, then add a pose to
`test_gestures.py` so it can't silently collide with an existing gesture.

---

## ⚙️ How it works

```mermaid
flowchart LR
    CAM([📹 Webcam]) --> GRAB["<b>CameraStream</b><br/>thread · newest frame only"]
    GRAB --> HANDS["<b>DetectWorker: hands</b><br/>HandLandmarker<br/>2 slots per person"]
    GRAB --> FACES["<b>DetectWorker: faces</b><br/>full-range detect<br/>→ per-face landmarks"]
    HANDS --> LOOP{{"<b>main loop</b>"}}
    FACES --> LOOP
    LOOP --> A["assign hands → faces"]
    A --> C["classify per person"]
    C --> D["composite meme GIF"]
    D --> OUT([🎥 Virtual camera])
    OUT --> APP([Zoom · Meet · Teams])

    classDef src fill:#845EF7,stroke:#5F3DC4,color:#fff
    classDef work fill:#00A6A6,stroke:#087F8C,color:#fff
    classDef main fill:#FF6B6B,stroke:#C92A2A,color:#fff
    classDef out fill:#51CF66,stroke:#2B8A3E,color:#fff
    class CAM,GRAB src
    class HANDS,FACES work
    class LOOP,A,C,D main
    class OUT,APP out
```

| file | role |
|:---|:---|
| `memecam.py` | capture loop, overlay placement, virtual-camera output, HUD |
| `detector.py` | MediaPipe Tasks wrappers + the tongue detector |
| `gesture_rules.py` | landmarks → gesture, person tracking, trigger debouncing |
| `overlay_fx.py` | sprite loading (GIF/WebP/PNG) and alpha compositing |
| `meme_assets.py` | gesture → meme file and how it's anchored |
| `fetch_gifs.py` | pulls the memes from Giphy |
| `fetch_models.py` | one-time MediaPipe model download (~13 MB) |
| `test_gestures.py` | synthetic-landmark regression tests (24 checks) |

<details>
<summary><b>🔍 Finding faces</b> — the bundled face detector finds 0 of 3 people, so it isn't used</summary>

<br>

MediaPipe's `FaceLandmarker` will happily run on the whole frame, but it uses a
**short-range** face detector internally, and that detector misses any face narrower than
about 10% of the frame width. Measured on test frames:

| people in frame | short-range (bundled) | full-range + crops |
|:---:|:---:|:---:|
| 2 | ✅ 2 / 2 | ✅ 2 / 2 |
| 3 | ❌ 0 / 3 | ✅ 3 / 3 |
| 4 | ❌ 0 / 4 | ✅ 4 / 4 |
| 5 | ❌ 0 / 5 | ✅ 5 / 5 |

Two people worked by luck. So faces are found with the **full-range** BlazeFace detector
(~3.8 ms for the whole frame) and then landmarked one padded crop at a time (~6.7 ms per
face). Each crop yields 478 landmarks plus 52 blendshapes.

</details>

<details>
<summary><b>👥 Telling people apart</b> — without identity tracking, two people swap memes</summary>

<br>

`assign_hands()` gives each hand to the face it belongs to, measuring distance in *face
widths*, so someone at the back of the room is judged on the same scale as someone leaning
into the lens. A hand whose nearest face already has two falls through to the next-best
face rather than being dropped.

`PersonTracker` then matches faces frame to frame by centre distance. MediaPipe returns an
unordered list with no identifiers, so without this two people would swap memes — and each
other's trigger state — whenever the detector happened to reorder its output.

</details>

<details>
<summary><b>🖐 Reading gestures</b> — thresholds in face widths, so distance doesn't matter</summary>

<br>

Every threshold in `gesture_rules.THRESH` is expressed as a multiple of **your face
width**, so the same numbers work whether you're leaning into the camera or sitting across
the room. Finger extension is tested against the wrist rather than by the y-axis, so it
survives a rotated hand.

Expressions come from blendshapes: `jawOpen` and `eyeWide` drive the horse stare, per-eye
`eyeBlink` drives the wink — as the *difference* between the two eyes, since a normal blink
shuts both — and gaze deflection drives the side eye.

There is **no `tongueOut` blendshape** among MediaPipe's 52, so the tongue is measured from
pixels. `detector.tongue_score()` computes how much of the mouth opening is bright,
saturated pink (tongue) rather than dark cavity or desaturated teeth, after normalising the
mouth crop to a fixed height and eroding the mask off the lips — the lips are themselves
bright red, so a nearly-shut mouth would otherwise score ~1.0. Calibrated on real photos:

| what | score |
|:---|:---:|
| tongue out | ≈ 0.55 |
| closed mouth | ≈ 0.02 |
| open mouth, no tongue | ≈ 0.00 |
| **threshold** | **0.35** |

</details>

<details>
<summary><b>🤯 Gestures that look alike</b> — why thumbs-up used to hijack mind-blown</summary>

<br>

`THUMBS_UP` used to fire on the way *into* `MIND_BLOWN`: as your hands come up, one of them
reads as thumb-only for a frame or two, and that hand genuinely satisfies the thumbs-up
rule. Three things stop it:

1. `MIND_BLOWN` is easier to hit — 3 extended fingers instead of 4, and the hands may start
   lower.
2. Single-hand poses are suppressed entirely while **both** hands are up by your head.
3. `THUMBS_UP` needs 4 consecutive frames instead of 2.

`DOUBLE_THUMBS` is checked before all of that, so two thumbs held high still work.

</details>

<details>
<summary><b>🎯 Not flickering</b> — every gesture is debounced, per person</summary>

<br>

`Stabilizer` requires a gesture to be held for a few frames before it fires, and to be
absent for several frames (plus a minimum hold time) before it releases. Gestures that are
easy to trigger by accident ask for more: the horse stare needs 12 frames, the wink 4 so it
outlasts an ordinary blink. Each person gets their own `Stabilizer`, and `FaceTracker`
smooths the face box *and* the expression signals while coasting 0.5 s through dropped
detections.

</details>

<details>
<summary><b>🖼 Drawing the overlay</b> — real alpha blending, and cover-fit sizing</summary>

<br>

`alpha_blend()` does `out = sprite·α + frame·(1−α)` with the sprite's own alpha channel,
clipped so an overlay can hang off the edge of the frame. Assigning
`frame[y:y+h, x:x+w] = sprite[:, :, :3]` instead is the classic mistake that turns
transparent pixels into a solid black box.

Overlays are **cover-fitted**: scaled so the meme covers your face box on *both* axes given
its own aspect ratio — `scale × max(face_w, face_h × sprite_w/sprite_h)`. Sizing on width
alone makes a tall portrait GIF tower over your face while a wide one sits across it like a
letterbox, and the mismatch gets worse the further you sit from the camera. Sprite resizes
are cached per 8-pixel width bucket under a 96 MB ceiling.

</details>

<details>
<summary><b>⚡ Speed</b> — threading takes five people from 15 fps to 58 fps</summary>

<br>

Profiled at 1280×720 on an M3 MacBook:

| stage | cost |
|:---|:---|
| hand landmarker | ~29 ms/frame (flat from 4 to 10 hand slots) |
| full-range face detector | ~3.8 ms/frame |
| face landmarks + blendshapes | ~6.7 ms **per person** |
| overlay compositing | ~3 ms/frame |

Five people in one thread is ~66 ms/frame — **15 fps**, so the video stutters and gestures
land late. Hands and faces therefore get a thread each, leaving the main loop with just
capture and compositing. Measured end-to-end over 200 frames:

| | inline | threaded |
|:---|:---:|:---:|
| 2 people | 18.0 fps | **57.5 fps** |
| 5 people | 14.6 fps | **58.2 fps** |

The main loop's cost no longer depends on how many people are in the shot. `CameraStream`
is the third thread: it drains the webcam and always serves the newest frame, because a
webcam keeps filling its queue while you work — without it, one slow frame leaves you
permanently behind, showing gestures from the past.

Two things that sound like optimisations but aren't, both measured and rejected:
**downscaling the detector input** (at 0.5× the face box jumped ~40 px and the inner-lip
landmarks became useless for the tongue test — it scored 1.000 on a *closed* mouth), and
**MediaPipe's GPU delegate** (crashes the process on this build).

</details>

---

## 🎚 Tuning

```bash
./run.sh --no-vcam --preview --debug
```

The HUD shows, per tracked person: their id, the current gesture, each hand's
finger-extension bitmask (thumb→pinky), and the live `jaw` / `wide` / `tongue` / `wink` /
`gaze` values — plus frame rate, measured trigger latency and how many stale camera frames
were dropped.

Any threshold can be overridden without editing code:

```bash
./run.sh --preview --thresh tongue=0.30 --thresh wink_gap=0.30 --thresh side_eye=0.35
```

| symptom | fix |
|:---|:---|
| 🔇 won't fire | hold the pose, read the live number off the HUD, set that threshold just under what you actually reach |
| 🔊 too eager | raise `--on-frames`, or tighten that threshold |
| 💨 drops out mid-pose | raise `--off-frames` / `--hold` |
| 🐢 want it snappier | `--fast` — fires on the first matching frame, ~33 ms at 30 fps instead of ~66 ms |
| 😛 tongue misfires from far away | raise `tongue`, or move closer; the score climbs on a tiny face because the mouth is only a few pixels tall |

Expression thresholds are the ones most worth tuning — blendshape response varies from
person to person.

> [!TIP]
> Run `python test_gestures.py` after changing any rule. It builds a synthetic hand skeleton
> for each gesture and checks the classifier returns it, that no pose steals another's
> overlay, that the mind-blown/thumbs-up collision stays fixed, that two people's hands
> don't get mixed up, and that identities survive the detector reordering its output.

---

## 🩺 Troubleshooting

| problem | what's going on |
|:---|:---|
| **"Could not open a virtual camera"** | Install OBS and click *Start Virtual Camera* once so the device gets registered, then re-run. Meanwhile `--no-vcam --preview` still works. |
| **Camera won't open on macOS** | Grant camera access to your terminal in System Settings → Privacy & Security → Camera. `tccutil reset Camera` re-triggers the prompt. |
| **`No module named 'mediapipe'`** | Check the Python version — MediaPipe needs 3.10–3.12. `./setup.sh` refuses anything else and says so. |
| **`module 'mediapipe' has no attribute 'solutions'`** | You're reading an older tutorial. MediaPipe 0.10.35+ removed the whole `mp.solutions.*` API and ships no bundled models; this project uses the current Tasks API, which is why models download separately. |
| **A gesture never fires** | Run with `--debug` and watch the HUD while you hold it — the finger bitmask and expression numbers show which condition is failing. |
| **Zoom doesn't list the virtual camera** | Start MemeCam before Zoom, and reload the tab for browser-based Meet/Teams. |

---

## 📝 Notes

- 🖥 Tested on **macOS (Apple Silicon)**. `pyvirtualcam` also supports OBS on Windows and
  v4l2loopback on Linux; everything except the virtual-camera backend is
  platform-independent, but those paths are untested here.
- 🎞 The meme GIFs are **not** included in this repository — they're other people's content.
  `fetch_gifs.py` downloads them from Giphy for personal use at setup time. Point it at
  whatever GIFs you have the right to use.
- 🔒 Everything runs locally. No video, image or landmark ever leaves your machine.

<div align="center">
<br>
<sub>Built with MediaPipe, OpenCV and pyvirtualcam.</sub>
</div>
