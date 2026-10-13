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

# Power check (warning only): an undervoltage on 2026-10-13 knocked the Hailo off the PCIe bus
# ("Device disconnected while opening device") until a reboot.
throttled=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
if [ -n "$throttled" ] && [ "$((throttled))" -ne 0 ]; then
    now=""; [ $((throttled & 0x1)) -ne 0 ] && now=" (undervoltage RIGHT NOW)"
    echo "WARNING: power problem since boot, throttled=$throttled$now. Undervoltage can make the Hailo drop out;" \
         "check the 5 V supply, and reboot if the Hailo fails to open." | tee -a "$log"
fi

python3 -u main_record.py "$@" 2>&1 | tee -i -a "$log"
