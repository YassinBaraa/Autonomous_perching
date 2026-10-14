#!/usr/bin/env bash
# Run ON THE LAPTOP (Linux, macOS, WSL or Git Bash): starts the vision with the live view on the drone's
# Pi over SSH and opens the live view in the browser -- everything at once. Copy it to the laptop:
#   scp crow-black@192.168.0.128:Autonomous_perching/UDP_client/laptop/perch_live.sh .
# Usage:   ./perch_live.sh [--manual] [user@pi-address]
# The terminal stays the vision's terminal (r reset, 1-9 branch, m manual, a automatic, Ctrl+C stops).
# The page goes through the SSH connection (localhost:8080), so no firewall port needs to be open.
PI="${PERCH_PI:-crow-black@192.168.0.128}"
PORT="${PERCH_PORT:-8080}"
OPTS=()
for arg in "$@"; do
    case "$arg" in
        --manual) OPTS+=(--manual) ;;
        *@*) PI="$arg" ;;
        *) echo "unknown option: $arg (use --manual, user@host)"; exit 2 ;;
    esac
done
URL="http://localhost:${PORT}"
(   # the page waits for the vision by itself, so it can open right away
    sleep 3
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
    elif command -v open >/dev/null 2>&1; then open "$URL"
    elif command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$URL"
    else echo "Open $URL in your browser"; fi
) >/dev/null 2>&1 &
echo "Starting the vision on $PI; live view at $URL"
exec ssh -t -L "${PORT}:localhost:${PORT}" "$PI" "~/Autonomous_perching/UDP_client/run_browser.sh --port=${PORT} ${OPTS[*]}"
