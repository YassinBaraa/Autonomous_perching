#!/usr/bin/env python3
"""Resource use of the vision pipeline on the Raspberry Pi 5: CPU (process and system), resident
memory, SoC temperature and SoC power (PMIC rails, sum of V x I), sampled every 0.25 s through three
phases: idle, the full pipeline replaying a recorded flight (detector while searching, KLT while
tracking), and the detector + post-processing running continuously. Uses the Hailo.
Writes paper/data/resources/samples.csv, modes.csv; summary in resources_summary.json."""
import csv
import os
import re
import subprocess
import sys
import time

DATA = "/home/crow-black/Autonomous_perching/paper/data/resources"
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource_child.py")
VIDEO = "/home/crow-black/Autonomous_perching/UDP_client/recordings/20261013_181122_raw.mp4"


def cpu_total():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]


def proc_cpu(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])
    except OSError:
        return None


def proc_rss_mb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return None


def temp_c():
    return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0


def soc_power_w():
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout
    amps = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\S+)_A current\(\d+\)=([\d.]+)A", out)}
    volts = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\S+)_V volt\(\d+\)=([\d.]+)V", out)}
    return sum(a * volts[k] for k, a in amps.items() if k in volts)


def flight_running():
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if argv and b"python" in argv[0].split(b"/")[-1] and any(a.endswith(b"main_record.py") for a in argv[1:3]):
            return True
    return False


def sample_phase(phase, pid, seconds, w, hz=4.0, stop=None):
    tick = os.sysconf("SC_CLK_TCK")
    t_end = time.time() + seconds
    last_tot, last_idle = cpu_total()
    last_p = proc_cpu(pid) if pid else None
    last_t = time.time()
    k = 0
    while time.time() < t_end:
        if stop is not None and stop():
            break
        time.sleep(1.0 / hz)
        now = time.time()
        tot, idle = cpu_total()
        sys_cpu = 100.0 * (1 - (idle - last_idle) / max(tot - last_tot, 1)) * os.cpu_count()
        pc = proc_cpu(pid) if pid else None
        p_cpu = 100.0 * (pc - last_p) / tick / (now - last_t) if pc is not None and last_p is not None else None
        pw = soc_power_w() if k % 2 == 0 else None
        w.writerow([phase, f"{now:.3f}", round(sys_cpu, 1), None if p_cpu is None else round(p_cpu, 1),
                    None if not pid else proc_rss_mb(pid), temp_c(), None if pw is None else round(pw, 3)])
        last_tot, last_idle, last_p, last_t = tot, idle, pc, now
        k += 1


def main():
    if flight_running():
        sys.exit("vision is running -- not measuring")
    os.makedirs(DATA, exist_ok=True)
    f = open(f"{DATA}/samples.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["phase", "wall", "sys_cpu_pct", "proc_cpu_pct", "rss_mb", "temp_c", "soc_power_w"])
    print("idle 20 s", flush=True)
    sample_phase("idle", None, 20, w)
    for phase, args in (("replay", ["replay", VIDEO]), ("detector_only", ["detector", VIDEO, "30"])):
        print(phase, flush=True)
        p = subprocess.Popen([sys.executable, CHILD, *args, f"{DATA}/modes_{phase}.csv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sample_phase(phase, p.pid, 600, w, stop=lambda: p.poll() is not None)
        p.wait()
        f.flush()
        time.sleep(5)
    f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
