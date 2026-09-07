#!/usr/bin/env bash
# One-shot setup: virtualenv, dependencies, MediaPipe models, meme GIFs.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.12 python3.11 python3.10 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    version=$("$candidate" -c 'import sys; print("%d%02d" % sys.version_info[:2])')
    if [ "$version" -ge 310 ] && [ "$version" -le 312 ]; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
    echo "! Need Python 3.10-3.12 (MediaPipe has no wheels for 3.13+, and macOS"
    echo "  ships 3.9 which is too old). On macOS:  brew install python@3.12"
    exit 1
fi

echo "==> Using $($PY --version) at $(command -v "$PY")"
[ -d .venv ] || "$PY" -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
echo "==> Installing dependencies (this pulls MediaPipe, ~100 MB)"
./.venv/bin/python -m pip install --quiet -r requirements.txt
echo "==> Downloading MediaPipe models"
./.venv/bin/python fetch_models.py
echo "==> Downloading meme GIFs from Giphy"
./.venv/bin/python fetch_gifs.py
./.venv/bin/python test_gestures.py >/dev/null && echo "==> Self-test passed"

cat <<'MSG'

Done. Start it with:

    ./run.sh --preview

Then pick "OBS Virtual Camera" as your camera in Zoom / Meet / Teams.
On macOS you also need OBS installed once, so the virtual camera device exists.
MSG
