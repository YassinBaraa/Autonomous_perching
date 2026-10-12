#!/usr/bin/env bash
# Start main_record.py and save everything it prints to recordings/run_<timestamp>.log
# (next to the videos of the same run). Works from any directory:
#   ~/Autonomous_perching/UDP_client/run_record.sh
# 'r' (reset target), 'q' and Ctrl+C work as usual; tee -i survives Ctrl+C so the
# shutdown messages still reach the log.
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")" || exit 1
mkdir -p recordings
log="recordings/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $PWD/$log"
python3 -u main_record.py "$@" 2>&1 | tee -i "$log"
