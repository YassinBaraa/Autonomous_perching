@echo off
rem Run ON A WINDOWS LAPTOP (Windows 10/11 have ssh built in): starts the vision with the live view
rem on the drone's Pi over SSH and opens the live view in the browser.
rem Usage:   perch_live.bat            or   perch_live.bat --manual
rem The window stays the vision's terminal (r reset, 1-9 branch, m manual, a automatic, Ctrl+C stops).
set PI=crow-black@192.168.0.128
set PORT=8080
start "" cmd /c "timeout /t 3 >nul & start http://localhost:%PORT%"
ssh -t -L %PORT%:localhost:%PORT% %PI% "~/Autonomous_perching/UDP_client/run_browser.sh --port=%PORT% %*"
