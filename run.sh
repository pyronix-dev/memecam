#!/bin/sh
# Convenience launcher: always uses the project venv.
cd "$(dirname "$0")" && exec ./.venv/bin/python memecam.py "$@"
