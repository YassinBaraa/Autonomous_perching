#!/usr/bin/env bash
# Like run_record.sh, plus the live view in a browser (camera, perch point, branch choice):
#   ~/Autonomous_perching/UDP_client/run_browser.sh            automatic branch choice
#   ~/Autonomous_perching/UDP_client/run_browser.sh --manual   nothing is locked until you choose a branch
# Then open http://<pi address>:8080 (printed below), or run laptop/perch_live.sh on the laptop,
# which starts this script over SSH and opens the browser by itself.
# Keys in this terminal: r reset, 1-9 choose a branch, m manual, a automatic, Ctrl+C stop.
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")" || exit 1
export LIVE_VIEW=1
for arg in "$@"; do
    case "$arg" in
        --manual) export LIVE_VIEW_MANUAL=1 ;;
        --port=*) export LIVE_VIEW_PORT="${arg#--port=}" ;;
        *) echo "unknown option: $arg (use --manual, --port=N)"; exit 2 ;;
    esac
done
port="${LIVE_VIEW_PORT:-8080}"
ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "================================================================"
echo " Live view: http://${ip:-<pi address>}:${port}"
echo " (from the laptop through SSH: ssh -L ${port}:localhost:${port} $(whoami)@${ip:-<pi>}  ->  http://localhost:${port})"
[ "${LIVE_VIEW_MANUAL:-0}" = 1 ] && echo " MANUAL: nothing is locked until you choose a branch"
echo "================================================================"
exec ./run_record.sh
