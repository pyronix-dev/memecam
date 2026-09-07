#!/usr/bin/env python3
"""Download the meme GIFs from Giphy into assets/.

    python fetch_gifs.py                       # fetch everything in GIPHY_LINKS
    python fetch_gifs.py --force               # re-download even if present
    python fetch_gifs.py SCUBA <giphy-url>     # point one gesture at another GIF

A Giphy page URL ends with the GIF id ("...-cat-scuba-HcKnldPBuyCyf5h2nY"), and
https://i.giphy.com/media/<id>/giphy.gif serves the raw animation - no API key.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request

from meme_assets import ASSET_DIR, GESTURE_STYLE

# gesture -> the Giphy page URL it came from (edit freely, then re-run)
GIPHY_LINKS = {
    "SCUBA":       "https://giphy.com/gifs/RespectiveCollective-cat-scuba-HcKnldPBuyCyf5h2nY",
    "MIND_BLOWN":  "https://giphy.com/gifs/stress-i-need-a-drink-brain-explode-2rqEdFfkMzXmo",
    "TONGUE":      "https://giphy.com/gifs/tongue-out-dog-a0yQSPMwtKPEC1QWSs",
    "HORSE":       "https://giphy.com/gifs/janmetternich-meme-horse-chromehearts-JMt40tiMr3LOKwkq0E",
    "HANDS_MOUTH": "https://giphy.com/gifs/jjk-jujutsu-kaisen-higuruma-JF1bGLHo6nY6t8LuFe",
    "THUMBS_UP":   "https://giphy.com/gifs/thumbs-up-111ebonMs90YLu",
    "TIMEOUT":     "https://giphy.com/gifs/Bounce-TV-wait-hold-on-time-out-k74OUg6bPJKy2JmyoS",
    "HEART":       "https://giphy.com/gifs/papajohns-shaq-shaquille-o-neal-TKL99l61UBNnnQSx4o",
    "DOUBLE_THUMBS": "https://giphy.com/gifs/shaun-the-sheep-movie-not-my-gif-2016-oscar-nominations-tIeCLkB8geYtW",
    "WINK":        "https://giphy.com/gifs/reaction-wrBURfbZmqqXu",
    "CALL_ME":     "https://giphy.com/gifs/dj-khaled-LSFXlAmuWhf6KN49FG",
    "SIDE_EYE":    "https://giphy.com/gifs/moodman-monkey-side-eye-sideeye-H5C8CevNMbpBqNqFjl",
}

MEDIA_URL = "https://i.giphy.com/media/{gif_id}/giphy.gif"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) memecam/1.0"}


def giphy_id(url_or_id: str) -> str:
    """Accept a full Giphy page URL, a media URL, or a bare id."""
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9]{6,}", url_or_id):
        return url_or_id
    m = re.search(r"/media/([A-Za-z0-9]+)/", url_or_id) or \
        re.search(r"giphy\.com/(?:gifs|clips|embed)/(?:[^/?#]*-)?([A-Za-z0-9]+)", url_or_id)
    if not m:
        raise ValueError(f"could not find a Giphy id in {url_or_id!r}")
    return m.group(1)


def download(gesture: str, url: str, asset_dir: str = ASSET_DIR, force: bool = False) -> str:
    if gesture not in GESTURE_STYLE:
        raise KeyError(f"unknown gesture {gesture!r} - known: {', '.join(GESTURE_STYLE)}")
    os.makedirs(asset_dir, exist_ok=True)
    dest = os.path.join(asset_dir, GESTURE_STYLE[gesture]["file"] + ".gif")
    if os.path.exists(dest) and not force:
        print(f"  [have] {gesture:<12} {os.path.basename(dest)}")
        return dest

    src = MEDIA_URL.format(gif_id=giphy_id(url))
    req = urllib.request.Request(src, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    if not data.startswith(b"GIF8"):
        raise ValueError(f"{src} did not return a GIF (got {data[:12]!r})")
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)
    print(f"  [ ok ] {gesture:<12} {os.path.basename(dest):<20} {len(data)/1e6:5.2f} MB")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="Download meme GIFs from Giphy")
    ap.add_argument("gesture", nargs="?", help="a single gesture to (re)fetch")
    ap.add_argument("url", nargs="?", help="its Giphy page URL")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    jobs = ({args.gesture.upper(): args.url} if args.gesture and args.url
            else GIPHY_LINKS)
    force = args.force or bool(args.gesture)

    failed = []
    for gesture, url in jobs.items():
        try:
            download(gesture, url, force=force)
        except Exception as exc:
            failed.append(gesture)
            print(f"  [fail] {gesture:<12} {exc}")
    if failed:
        print(f"\n! failed: {', '.join(failed)}")
        return 1
    print(f"\nGIFs in {ASSET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
