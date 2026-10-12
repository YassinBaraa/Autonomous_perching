#!/usr/bin/env bash
# Draw the overlay video of a recorded run (after the flight):
#   ~/Autonomous_perching/UDP_client/render_overlay.sh            # the newest run
#   ~/Autonomous_perching/UDP_client/render_overlay.sh recordings/20261012_114345   # a given run
# Reads <run>_raw.mp4 + <run>_overlay.jsonl, writes <run>_ibvs_overlay.mp4.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
exec python3 overlay.py "$@"
