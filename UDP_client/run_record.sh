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

# Power/heat check (warning only): an undervoltage on 2026-10-13 knocked the Hailo off the PCIe bus
# ("Device disconnected while opening device") until a reboot. vcgencmd get_throttled bits: 0x1/0x10000
# undervoltage now/since boot, 0x2/0x20000 frequency capped, 0x4/0x40000 throttled, 0x8/0x80000 soft
# temperature limit.
throttled=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
if [ -n "$throttled" ] && [ "$((throttled))" -ne 0 ]; then
    t=$((throttled))
    if [ $((t & 0x10001)) -ne 0 ]; then
        now=""; [ $((t & 0x1)) -ne 0 ] && now=" RIGHT NOW"
        echo "WARNING: undervoltage$now (throttled=$throttled). It can make the Hailo drop out:" \
             "check the 5 V supply, and reboot if the Hailo fails to open." | tee -a "$log"
    fi
    if [ $((t & 0x80008)) -ne 0 ]; then
        now=""; [ $((t & 0x8)) -ne 0 ] && now=" RIGHT NOW"
        echo "WARNING: the Pi got too hot$now (soft temperature limit, throttled=$throttled): the CPU" \
             "is slowed down and frames take longer; let it cool or improve cooling." | tee -a "$log"
    elif [ $((t & 0x10001)) -eq 0 ]; then
        echo "WARNING: the Pi was throttled since boot (throttled=$throttled)." | tee -a "$log"
    fi
fi

python3 -u main_record.py "$@" 2>&1 | tee -i -a "$log"
