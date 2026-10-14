"""Live view of the vision in a browser, and branch selection.

Open http://<pi address>:8080 on a computer in the same network. If that port is not reachable,
forward it through the SSH session instead:
    ssh -L 8080:localhost:8080 <user>@<pi address>      then open http://localhost:8080
(laptop/perch_live.sh does both and starts the vision.)

The page shows the camera image (640x360, up to 10 frames per second) with the perch point, the
stored target and, while searching, every detected branch: tinted by what can be done with it and
numbered. Click a branch (or press its number, in the page or in the terminal) to make the vision lock
on that branch; 'a' returns to the automatic choice, 'm' (manual) waits for your choice before
locking, 'r' drops the target and the branch choice (manual: choose again; automatic: the vision picks
again). A lost target is found again on the chosen branch. Branches that drop out of the detection for
a moment (motion) stay shown and clickable for SEEN_KEEP_S. LIVE_VIEW_MANUAL=1 starts in manual mode.

Cost for the flight loop: publish() and apply_commands() only store references and read a few
values (microseconds). Drawing and JPEG encoding run in a separate thread, and only while at least
one browser is connected. Off unless LIVE_VIEW=1 (run_browser.sh sets it; run_record.sh does not);
change the port with LIVE_VIEW_PORT.
"""
import http.server
import json
import os
import socket
import threading
import time

import cv2
import numpy as np

LIVE_FPS = 10.0
STREAM_SCALE = 0.5            # 1280x720 -> 640x360
JPEG_QUALITY = 72
SEEN_KEEP_S = 1.5             # a branch missing from the detection stays shown and clickable this long
CLICK_RADIUS_PX = 80          # [frame px] a click this close to a shown branch's mask chooses it
PENDING_RADIUS_PX = 120       # [frame px] a click on no shown branch: the first branch detected this close
CANDIDATES_SHOW_S = 1.0       # nothing is listed when the detection has not run for this long (tracking)

# Overlay colours (BGR) -- the page uses the same ones
COL = {"choosable": (255, 200, 0), "auto": (0, 196, 255), "chosen": (200, 64, 255), "end": (0, 128, 255),
       "off": (150, 150, 150), "point": (60, 60, 255), "stored": (255, 255, 255)}

_instance = None


def current():
    """The running LiveView, or None."""
    return _instance


def start(port=None):
    """Start the server (once) if LIVE_VIEW=1. Returns the LiveView, or None if off or the port is
    busy; never raises, so a problem here can not stop a flight."""
    global _instance
    if _instance is not None:
        return _instance
    if os.environ.get("LIVE_VIEW", "0") != "1":
        return None
    port = int(port or os.environ.get("LIVE_VIEW_PORT", 8080))
    try:
        lv = LiveView(port)
    except OSError as e:
        print(f"Live view NOT started (port {port}: {e}); the flight is not affected")
        return None
    _instance = lv
    if os.environ.get("LIVE_VIEW_MANUAL", "0") == "1":
        lv.manual = True
        lv.selection = _waiting()
        print("Live view: MANUAL -- nothing is locked until a branch is chosen")
    print(f"Live view: http://{_address()}:{port}   (or: ssh -L {port}:localhost:{port} <pi>, then http://localhost:{port})")
    print("  click a branch or press its number to choose it, 'm' wait for a choice, 'a' automatic, 'r' reset")
    return lv


def _address():
    try:   # no packet is sent: connect() on a UDP socket only picks the outgoing interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.0.1", 9))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostname()


def _waiting():
    """Manual mode before a branch is chosen: nothing is locked."""
    return {"track_id": None, "xy": None, "n": None}


def _why(c):
    if c["status"] == "direction":
        return "not vertical" + (f", {c['angle']:.0f} deg" if c.get("angle") is not None else "")
    if c["status"] == "end":
        return "too close to its end" + ("" if c["choosable"] else ", no point at all")
    return "ok"


def _r(p, nd=1):
    return None if p is None else [round(float(v), nd) for v in np.asarray(p).reshape(-1)[:2]]


def _mask_distance(mask, x, y, r):
    """Distance [px] from (x, y) to the nearest pixel of `mask` within r, or None."""
    if mask is None:
        return None
    h, w = mask.shape[:2]
    x, y = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, h), max(x - r, 0), min(x + r + 1, w)
    if y1 <= y0 or x1 <= x0:
        return None
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if not len(xs):
        return None
    d = float(np.min(np.hypot(xs + x0 - x, ys + y0 - y)))
    return d if d <= r else None


class LiveView:
    def __init__(self, port):
        self._lock = threading.Lock()
        self._frame = None          # latest camera frame (reference; never modified here)
        self._info = dict(point=None, stored=None, tstate=None, ctrl=None, size=None, rate=None, dir=None,
                          width=None, events=[], t=None)   # what is drawn on the frame
        self._seq = 0
        # Branches of the recent detection frames, by track: {track_id: entry}; an entry keeps its
        # number while the track is seen, and stays (stale) for SEEN_KEEP_S after it was last seen
        self._seen = {}
        self._numbers = {}          # track_id -> number shown
        self._cand_t = 0.0          # last detection frame
        self._commands = []
        self.selection = None       # None = automatic choice; else {"track_id", "xy", "n"} (all None: waiting)
        self.manual = False         # manual mode: nothing is locked until a branch is chosen
        self._jpeg, self._jpeg_seq = None, 0
        self._jpeg_cond = threading.Condition()
        self._clients = 0
        self._wake = threading.Event()
        self.view_fps = 0.0
        self._server = http.server.ThreadingHTTPServer(("0.0.0.0", port), _handler(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="live-view-http", daemon=True).start()
        threading.Thread(target=self._render_loop, name="live-view-render", daemon=True).start()

    # ------------------------------------------------------------------ flight loop side
    def publish(self, ctx, pipeline, packet, ctrl_state, rate_hz=None):
        """Once per frame from the main loop: the frame and the values to draw (references only)."""
        tr = getattr(pipeline, "tracker", None)
        st = getattr(pipeline, "size_tracker", None)
        tstate = None if tr is None else tr.state
        stored = None
        off, seen = getattr(pipeline, "model_offset", None), getattr(pipeline, "model_seen_t", None)
        if off is not None and seen is not None and time.monotonic() - seen <= 1.0 and tr is not None and tr.point is not None:
            stored = np.asarray(tr.point, dtype=float) + np.asarray(off, dtype=float)
        pt = ctx.estimated_point
        info = dict(point=_r(pt), stored=_r(stored), tstate=tstate, ctrl=ctrl_state,
                    size=None if ctx.target_size is None else round(float(ctx.target_size), 1),
                    rate=None if rate_hz is None else round(float(rate_hz), 1),
                    dir=None if st is None or st.direction is None or pt is None else _r(st.direction, 3),
                    width=None if st is None or not st.width or pt is None else round(float(st.width), 1),
                    events=[e[0] for e in ctx.debug.get("events", [])], t=time.monotonic())
        with self._lock:
            self._frame = ctx.frame
            self._info = info
            self._seq += 1
            # while a branch is chosen, its anchor follows the tracked point: after a loss the
            # detection then picks the same branch again at its current place
            sel = self.selection
            if sel is not None and (sel.get("track_id") is not None or sel.get("xy") is not None) \
                    and tstate == "tracking" and pt is not None:   # a chosen branch, not manual waiting
                sel["xy"] = (float(pt[0]), float(pt[1]))

    def apply_commands(self, pipeline, ctx):
        """Once per frame from the main loop: carry out what the page / keys asked for."""
        with self._lock:
            cmds, self._commands = self._commands, []
        for cmd in cmds:
            kind = cmd[0]
            if kind == "reset":
                self._reset(pipeline, ctx, "reset from the live view")
                self.clear_choice()
            elif kind == "auto":
                with self._lock:
                    self.selection, self.manual = None, False
                print("[live] automatic branch choice")
            elif kind == "manual":
                with self._lock:
                    self.selection, self.manual = _waiting(), True
                print("[live] manual: nothing is locked until a branch is chosen")
                if self._holds_target(pipeline):
                    self._reset(pipeline, ctx, "manual choice requested in the live view")
            elif kind in ("click", "number"):
                with self._lock:
                    if kind == "click":
                        e, why = self._entry_at(cmd[1], cmd[2])
                        if e is not None:
                            self.selection = {"track_id": e["track_id"], "xy": (e["x"], e["y"]), "n": e["n"]}
                            what = f"branch {e['n']}" + (" (seen just before)" if e["stale"] else "")
                        elif why:
                            print(f"[live] click at ({cmd[1]:.0f}, {cmd[2]:.0f}): {why}")
                            continue
                        else:   # no branch shown there: the first one detected near the click
                            self.selection = {"track_id": None, "xy": (cmd[1], cmd[2]), "n": None}
                            what = f"the first branch detected near ({cmd[1]:.0f}, {cmd[2]:.0f})"
                    else:
                        e = next((e for e in self._seen.values() if e["n"] == cmd[1]), None)
                        if e is None:
                            print(f"[live] no branch number {cmd[1]} shown")
                            continue
                        if not e["choosable"]:
                            print(f"[live] branch {cmd[1]} can not be chosen ({_why(e)})")
                            continue
                        self.selection = {"track_id": e["track_id"], "xy": (e["x"], e["y"]), "n": e["n"]}
                        what = f"branch {e['n']}"
                print(f"[live] {what} chosen")
                if self._holds_target(pipeline):
                    # a choice made while a target is held (or a lost one is searched for) means: drop it
                    # and lock on the chosen branch
                    self._reset(pipeline, ctx, f"{what} chosen in the live view")

    @staticmethod
    def _holds_target(pipeline):
        tr = getattr(pipeline, "tracker", None)
        tm = getattr(pipeline, "target_model", None)
        return (tr is not None and tr.state != "idle") or (tm is not None and tm.active)

    def _entry_at(self, x, y):
        """(entry, None) for the shown branch nearest a click (within CLICK_RADIUS_PX, choosable ones
        first); (None, reason) if only a branch that can not be chosen is there; (None, None) if none."""
        best, best_d, blocked = None, None, None
        for e in self._seen.values():
            d = _mask_distance(e.get("mask"), x, y, CLICK_RADIUS_PX)
            if d is None:
                continue
            if not e["choosable"]:
                blocked = blocked or f"branch {e['n']} can not be chosen ({_why(e)})"
                continue
            if best_d is None or d < best_d:
                best, best_d = e, d
        return (best, None) if best is not None else (None, blocked)

    def clear_choice(self):
        """After a reset ('r', the page's Reset): forget the chosen branch -- in manual mode wait for a
        new choice, in automatic mode the vision picks again."""
        with self._lock:
            self.selection = _waiting() if self.manual else None
        print("[live] branch choice cleared: " + ("choose a branch" if self.manual else "automatic choice"))

    @staticmethod
    def _reset(pipeline, ctx, why):
        ctx.estimated_point = None
        pipeline.reset_target(("RESET", why))
        print(f"[live] target dropped ({why}), searching")

    def command(self, *cmd):
        """From the HTTP thread or the terminal key reader."""
        with self._lock:
            self._commands.append(cmd)

    # ------------------------------------------------------------------ detection side
    def choose(self, dctx, best, mask):
        """From the detection loop: remember the branches (for drawing, numbers and clicks) and return
        (candidate, mask, chosen) to lock on: the automatic best (chosen False), or the operator's
        branch (chosen True). A branch rejected only for lack of room at its end ("end") is used when
        chosen explicitly. While the chosen branch is not detected, (None, None, False)."""
        now = time.monotonic()
        cands = list(dctx.candidates or [])
        rejected = list((getattr(dctx, "debug", None) or {}).get("rejected_candidates") or [])
        masks = {tid: m for _, m, _, _, tid, _ in (dctx.tracks or []) if tid is not None}
        best_tid = None if best is None else best.get("track_id")
        usable = cands + [r for r in rejected if r["reason"] == "end" and "x" in r]
        with self._lock:
            for c in [dict(c, status="ok") for c in cands] + [dict(r, status=r["reason"]) for r in rejected]:
                tid = c["track_id"]
                if tid not in self._numbers:
                    taken = set(self._numbers.values())
                    self._numbers[tid] = next(n for n in range(1, 100) if n not in taken)
                x1, y1, x2, y2 = (int(v) for v in c["bbox"])
                self._seen[tid] = dict(
                    n=self._numbers[tid], track_id=tid, status=c["status"],
                    choosable=c["status"] == "ok" or (c["status"] == "end" and "x" in c),
                    x=int(c["x"]) if "x" in c else (x1 + x2) // 2, y=int(c["y"]) if "y" in c else (y1 + y2) // 2,
                    has_point="x" in c, confidence=round(float(c.get("confidence") or 0.0), 3), bbox=[x1, y1, x2, y2],
                    angle=None if c.get("angle_deg") is None else round(float(c["angle_deg"]), 0),
                    auto=tid == best_tid, mask=masks.get(tid), t=now, stale=False, cand=c)
            for tid, e in list(self._seen.items()):
                if e["t"] < now:
                    e["stale"] = True
                    if now - e["t"] > SEEN_KEEP_S:
                        del self._seen[tid]
                        self._numbers.pop(tid, None)
            self._cand_t = now
            sel = None if self.selection is None else dict(self.selection)
        if sel is None:
            return best, mask, False
        if sel.get("track_id") is None and sel.get("xy") is None:
            return None, None, False                       # manual, nothing chosen yet
        chosen = None
        if sel.get("track_id") is not None:
            chosen = next((c for c in usable if c["track_id"] == sel["track_id"]), None)
            if chosen is None:
                with self._lock:
                    if sel["track_id"] in self._seen:          # missing for a moment: wait for it
                        return None, None, False
        if chosen is None and sel.get("xy") is not None:   # gone (or a click on nothing shown): nearest
            near = [(d, c) for c in usable
                    for d in [_mask_distance(masks.get(c["track_id"]), sel["xy"][0], sel["xy"][1], PENDING_RADIUS_PX)]
                    if d is not None]
            chosen = min(near, key=lambda dc: dc[0])[1] if near else None
        if chosen is None:
            return None, None, False
        with self._lock:
            if self.selection is not None:
                self.selection.update(track_id=chosen["track_id"], xy=(float(chosen["x"]), float(chosen["y"])),
                                      n=self._numbers.get(chosen["track_id"]))
        return chosen, masks.get(chosen["track_id"]), True

    # ------------------------------------------------------------------ server side
    def _branches(self):
        """Shown branches (no masks), for the state and drawing; caller holds the lock."""
        if time.monotonic() - self._cand_t > CANDIDATES_SHOW_S:
            return []
        sel_tid = None if self.selection is None else self.selection.get("track_id")
        out = []
        for e in sorted(self._seen.values(), key=lambda e: e["n"]):
            out.append({k: e[k] for k in ("n", "track_id", "status", "choosable", "x", "y", "has_point", "confidence",
                                          "bbox", "angle", "auto", "stale")} | {"chosen": e["track_id"] == sel_tid})
        return out

    def state(self):
        with self._lock:
            info, sel, frame = dict(self._info), self.selection, self._frame
            branches = self._branches()
        return dict(info, candidates=branches, frame=None if frame is None else [frame.shape[1], frame.shape[0]],
                    selection=None if sel is None else {"n": sel.get("n"), "xy": _r(sel.get("xy")),
                                                        "waiting": sel.get("xy") is None and sel.get("track_id") is None},
                    view_fps=round(self.view_fps, 1), clients=self._clients, manual=self.manual)

    def _render_loop(self):
        last_seq, t_last, n, t_fps = -1, 0.0, 0, time.monotonic()
        while True:
            if self._clients == 0:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            dt = 1.0 / LIVE_FPS - (time.monotonic() - t_last)
            if dt > 0:
                time.sleep(dt)
            with self._lock:
                frame, seq = self._frame, self._seq
                masks = {e["track_id"]: e["mask"] for e in self._seen.values()}
            if frame is None or (seq == last_seq and time.monotonic() - t_last < 1.0):
                time.sleep(0.02)
                continue
            t_last, last_seq = time.monotonic(), seq
            try:
                jpeg = self._render(frame, self.state(), masks)
            except Exception as e:   # never let drawing kill the thread
                print(f"[live] drawing failed: {e}")
                continue
            with self._jpeg_cond:
                self._jpeg, self._jpeg_seq = jpeg, self._jpeg_seq + 1
                self._jpeg_cond.notify_all()
            n += 1
            if time.monotonic() - t_fps >= 2.0:
                self.view_fps, n, t_fps = n / (time.monotonic() - t_fps), 0, time.monotonic()

    @staticmethod
    def _colour(b, auto_mode):
        if b["chosen"]:
            return COL["chosen"]
        if not b["choosable"]:
            return COL["off"]
        if b["status"] == "end":
            return COL["end"]
        return COL["auto"] if (auto_mode and b["auto"]) else COL["choosable"]

    def _render(self, frame, s, masks):
        k = STREAM_SCALE
        img = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        P = lambda p: (int(round(p[0] * k)), int(round(p[1] * k)))
        tracking = s.get("tstate") == "tracking" and s.get("point") is not None
        auto_mode = s.get("selection") is None
        if not tracking and s.get("candidates"):
            tint = img.copy()
            outlines = []
            for b in s["candidates"]:
                m = masks.get(b["track_id"])
                if m is None:
                    continue
                ms = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                col = self._colour(b, auto_mode)
                tint[ms > 0] = col
                cnts, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                outlines.append((cnts, col, 1 if b["stale"] else 2))
            img = cv2.addWeighted(tint, 0.38, img, 0.62, 0)
            for cnts, col, th in outlines:
                cv2.drawContours(img, cnts, -1, col, th, cv2.LINE_AA)
            for b in s["candidates"]:
                col = self._colour(b, auto_mode)
                at = P((b["x"], b["y"]))
                if b["has_point"] and b["choosable"]:
                    cv2.circle(img, at, 6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(img, at, 6, col, 1, cv2.LINE_AA)
                c = (min(max(at[0] + 16, 13), w - 13), min(max(at[1], 35), h - 13))   # badge kept inside the image
                cv2.circle(img, c, 11, col, -1, cv2.LINE_AA)
                cv2.circle(img, c, 11, (255, 255, 255) if not b["stale"] else (90, 90, 90), 1, cv2.LINE_AA)
                txt = str(b["n"])
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.putText(img, txt, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
            if s.get("selection") is not None and (s["selection"] or {}).get("n") is None and s["selection"].get("xy") is not None:
                cv2.drawMarker(img, P(s["selection"]["xy"]), COL["chosen"], cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 3)   # image centre
        cv2.drawMarker(img, (w // 2, h // 2), (40, 40, 40), cv2.MARKER_CROSS, 16, 1)
        if s.get("stored") is not None:
            cv2.circle(img, P(s["stored"]), 7, COL["stored"], 2, cv2.LINE_AA)
        if tracking:
            p = P(s["point"])
            if s.get("dir") is not None:
                d = np.asarray(s["dir"], float)
                a, b2 = np.asarray(p) - 34 * d, np.asarray(p) + 34 * d
                cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b2), (90, 230, 120), 2, cv2.LINE_AA)
            cv2.circle(img, p, 13, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(img, p, 13, COL["point"], 2, cv2.LINE_AA)
            cv2.drawMarker(img, p, COL["point"], cv2.MARKER_CROSS, 9, 1)
        # a slim status line for direct viewers of /stream.mjpg (the page shows its own)
        mode = "TRACKING" if tracking else ("LOST" if s.get("tstate") == "lost" else "SEARCHING")
        sel = s.get("selection")
        choice = "AUTO" if sel is None else ("MANUAL: CHOOSE A BRANCH" if sel.get("waiting") else
                                             (f"BRANCH {sel.get('n')}" if sel.get("n") else "WAITING FOR THE BRANCH"))
        parts = [mode, s.get("ctrl") or "no ctrl", f"size {s['size']:.1f}px" if s.get("size") is not None else "", choice]
        text = "   ".join(p for p in parts if p)
        bar = img[0:22].copy()
        cv2.rectangle(bar, (0, 0), (w, 22), (0, 0, 0), -1)
        img[0:22] = cv2.addWeighted(bar, 0.55, img[0:22], 0.45, 0)
        cv2.putText(img, text, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def jpeg_seq(self):
        with self._jpeg_cond:
            return self._jpeg_seq

    def next_jpeg(self, after_seq, timeout=2.0):
        with self._jpeg_cond:
            self._jpeg_cond.wait_for(lambda: self._jpeg_seq > after_seq, timeout=timeout)
            return self._jpeg, self._jpeg_seq

    def client(self, delta):
        with self._lock:
            self._clients += delta
        if delta > 0:
            self._wake.set()


def _handler(lv):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):   # nothing in the flight log
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(lv.state()).encode(), "application/json")
            elif path == "/snapshot.jpg":   # the page fetches frames one by one (works in every browser)
                lv.client(+1)
                try:
                    jpeg, _ = lv.next_jpeg(lv.jpeg_seq(), timeout=2.0)   # a frame drawn after this request
                finally:
                    lv.client(-1)
                if jpeg:
                    self._send(200, jpeg, "image/jpeg")
                else:
                    self._send(503, b"no frame yet", "text/plain")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                lv.client(+1)
                seq = 0
                try:
                    while True:
                        jpeg, seq2 = lv.next_jpeg(seq)
                        if jpeg is None or seq2 == seq:
                            continue
                        seq = seq2
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                         str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    pass
                finally:
                    lv.client(-1)
                    self.close_connection = True
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            path = self.path.split("?")[0]
            if path == "/api/select":
                fw, fh = (lv.state().get("frame") or [1280, 720])
                if "n" in body:
                    lv.command("number", int(body["n"]))
                elif "u" in body and "v" in body:   # click position as a fraction of the image
                    lv.command("click", float(body["u"]) * fw, float(body["v"]) * fh)
                else:
                    self._send(400, b"need n or u,v", "text/plain")
                    return
            elif path == "/api/auto":
                lv.command("auto")
            elif path == "/api/manual":
                lv.command("manual")
            elif path == "/api/reset":
                lv.command("reset")
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, b"{}", "application/json")
    return Handler


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Perching vision</title>
<style>
/* Control console for a flying camera: dark, quiet chrome; colour only where it carries meaning (the
   same colours as the overlay on the image). */
:root {
  --bg:#0b0e11; --panel:#141a1f; --panel2:#1b232a; --line:#26313a; --ink:#e8eef2; --muted:#8e9ca7;
  --choosable:#00c8ff; --auto:#ffc400; --chosen:#ff40c8; --end:#ff8000; --off:#969696; --point:#ff3c3c;
  --ok:#3ddc84; --warn:#ffc400; --bad:#ff5a5a;
  --radius:12px; --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Inter, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  color-scheme: dark;
}
* { box-sizing:border-box; }
html, body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 var(--font); }
.app { max-width:1400px; margin:0 auto; padding:14px 16px 22px; display:grid; gap:14px; }
header { display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px; }
.brand { display:flex; align-items:center; gap:10px; font-weight:650; font-size:16px; letter-spacing:.01em; }
.conn { width:9px; height:9px; border-radius:50%; background:var(--bad); box-shadow:0 0 0 3px rgba(255,90,90,.15); }
.conn.on { background:var(--ok); box-shadow:0 0 0 3px rgba(61,220,132,.15); }
.pills { display:flex; flex-wrap:wrap; gap:8px; margin-left:auto; }
.pill { font:600 12px/1 var(--mono); letter-spacing:.04em; padding:7px 10px; border-radius:999px; background:var(--panel);
        border:1px solid var(--line); color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
.pill.state-tracking { color:#08140d; background:var(--ok); border-color:var(--ok); }
.pill.state-searching { color:#1a1400; background:var(--warn); border-color:var(--warn); }
.pill.state-lost { color:#1a0505; background:var(--bad); border-color:var(--bad); }
main { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:14px; align-items:start; }
@media (max-width: 900px) { main { grid-template-columns:1fr; } }
.stage { position:relative; background:#000; border:1px solid var(--line); border-radius:var(--radius); overflow:hidden;
         box-shadow:0 10px 30px rgba(0,0,0,.35); }
.stage img { display:block; width:100%; height:auto; aspect-ratio:16/9; cursor:crosshair; background:#000; user-select:none; }
#offline { position:absolute; inset:0; display:grid; place-items:center; background:rgba(0,0,0,.55); color:var(--muted);
           pointer-events:none; font-size:15px; }
#offline[hidden] { display:none; }
#marks { position:absolute; inset:0; pointer-events:none; }
.ripple { position:absolute; width:46px; height:46px; margin:-23px 0 0 -23px; border-radius:50%; border:2px solid var(--chosen);
          animation:ripple .7s ease-out forwards; }
@keyframes ripple { from { transform:scale(.3); opacity:1; } to { transform:scale(1.4); opacity:0; } }
#toast { position:absolute; left:50%; top:14px; transform:translateX(-50%); background:rgba(20,26,31,.92); border:1px solid var(--line);
         color:var(--ink); padding:8px 14px; border-radius:999px; font-weight:600; font-size:13px; opacity:0; transition:opacity .2s;
         pointer-events:none; white-space:nowrap; }
#toast.show { opacity:1; }
aside { display:grid; gap:12px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:14px; display:grid; gap:10px; }
.card h2 { margin:0; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:650; }
.seg { display:grid; grid-template-columns:1fr 1fr; background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:3px; gap:3px; }
.seg button { border:0; background:transparent; color:var(--muted); padding:9px 10px; border-radius:8px; font:600 13px var(--font); cursor:pointer; }
.seg button.on { background:var(--ink); color:#0b0e11; }
.seg button:focus-visible, .btn:focus-visible, .branch:focus-visible { outline:2px solid var(--choosable); outline-offset:2px; }
.btn { width:100%; display:flex; justify-content:center; align-items:center; gap:8px; padding:10px 12px; border-radius:10px;
       border:1px solid var(--line); background:var(--panel2); color:var(--ink); font:600 13px var(--font); cursor:pointer; }
.btn:hover { border-color:var(--muted); }
kbd { font:600 11px var(--mono); padding:1px 6px; border-radius:5px; border:1px solid var(--line); color:var(--muted); background:var(--bg); }
.seg button.on kbd { background:transparent; color:#0b0e11; border-color:rgba(0,0,0,.25); }
#branches { list-style:none; margin:0; padding:0; display:grid; gap:6px; }
.branch { width:100%; display:grid; grid-template-columns:28px 1fr auto; align-items:center; gap:10px; text-align:left;
          padding:8px 10px; border-radius:10px; border:1px solid var(--line); background:var(--panel2); color:var(--ink);
          font:500 13px var(--font); cursor:pointer; }
.branch:hover:not(:disabled) { border-color:var(--muted); }
.branch:disabled { cursor:not-allowed; opacity:.55; }
.branch.stale { opacity:.6; }
.branch.chosen { border-color:var(--chosen); box-shadow:inset 0 0 0 1px var(--chosen); }
.num { width:26px; height:26px; border-radius:50%; display:grid; place-items:center; font:700 12px var(--mono); color:#0b0e11; }
.why { color:var(--muted); font-size:12px; }
.conf { font:600 12px var(--mono); color:var(--muted); font-variant-numeric:tabular-nums; }
.empty { color:var(--muted); font-size:13px; padding:4px 2px; }
.legend { display:grid; grid-template-columns:1fr 1fr; gap:6px 10px; font-size:12px; color:var(--muted); }
.legend span::before { content:""; display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; background:var(--c); }
.help { margin:0; color:var(--muted); font-size:12.5px; }
</style></head><body><div class="app">
<header>
  <div class="brand"><span class="conn" id="conn" title="connection to the vision"></span>Perching vision</div>
  <div class="pills">
    <span class="pill" id="p-mode">—</span><span class="pill" id="p-ctrl">controller —</span>
    <span class="pill" id="p-size">size —</span><span class="pill" id="p-rate">— Hz</span><span class="pill" id="p-view">view —</span>
  </div>
</header>
<main>
  <div class="stage" id="stage"><img id="cam" alt="Live camera image with the detected branches and the perch point">
    <div id="marks"></div><div id="toast"></div><div id="offline">waiting for the camera image…</div></div>
  <aside>
    <div class="card">
      <h2>Branch choice</h2>
      <div class="seg" role="group" aria-label="Branch choice">
        <button id="auto" title="The vision picks the branch">Auto <kbd>A</kbd></button>
        <button id="manual" title="Nothing is locked until you choose a branch">Manual <kbd>M</kbd></button>
      </div>
      <button class="btn" id="reset" title="Drop the target and the branch choice">Reset target <kbd>R</kbd></button>
    </div>
    <div class="card">
      <h2>Branches</h2>
      <ul id="branches"><li class="empty">—</li></ul>
      <div class="legend">
        <span style="--c:var(--choosable)">can be chosen</span><span style="--c:var(--auto)">automatic pick</span>
        <span style="--c:var(--chosen)">your choice</span><span style="--c:var(--end)">too close to its end</span>
        <span style="--c:var(--off)">not vertical</span><span style="--c:var(--point)">perch point</span>
      </div>
    </div>
    <p class="help">Click a branch in the image or press its number. A branch that drops out for a moment while
      you move stays shown (faded) and can still be clicked. <kbd>R</kbd> drops the target and the choice
      (manual: choose again). Orange: choosing it overrides the end rule.</p>
  </aside>
</main>
</div>
<script>
const $ = id => document.getElementById(id);
const cam = $('cam'), offline = $('offline'), stage = $('stage');
const COLORS = {choosable:'var(--choosable)', auto:'var(--auto)', chosen:'var(--chosen)', end:'var(--end)', off:'var(--off)'};
let lastFrame = 0, prevUrl = null, last = {}, toastTimer = null;

// camera frames, one at a time (works in every browser)
async function frameLoop() {
  try {
    const r = await fetch('/snapshot.jpg?' + Date.now(), {cache: 'no-store'});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const url = URL.createObjectURL(await r.blob());
    cam.src = url;
    if (prevUrl) URL.revokeObjectURL(prevUrl);
    prevUrl = url; lastFrame = Date.now(); offline.hidden = true;
    setTimeout(frameLoop, 0);
  } catch (e) {
    offline.textContent = 'no camera image (vision not running?) — retrying'; offline.hidden = false;
    setTimeout(frameLoop, 1000);
  }
}
frameLoop();
setInterval(() => { if (Date.now() - lastFrame > 3000) { offline.textContent = 'no new camera image for 3 s — retrying'; offline.hidden = false; } }, 1000);

function post(path, body) {
  return fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body || {})}).catch(() => {});
}
function toast(text) {
  const t = $('toast'); t.textContent = text; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 1600);
}
cam.addEventListener('click', e => {
  const r = cam.getBoundingClientRect();
  const u = (e.clientX - r.left) / r.width, v = (e.clientY - r.top) / r.height;
  const m = document.createElement('div'); m.className = 'ripple';
  m.style.left = (u * 100) + '%'; m.style.top = (v * 100) + '%';
  $('marks').appendChild(m); setTimeout(() => m.remove(), 800);
  post('/api/select', {u, v});
});
$('auto').onclick = () => { post('/api/auto'); toast('Automatic branch choice'); };
$('manual').onclick = () => { post('/api/manual'); toast('Manual: choose a branch'); };
$('reset').onclick = () => { post('/api/reset'); toast('Target dropped'); };
document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key >= '1' && e.key <= '9') post('/api/select', {n: +e.key});
  else if (e.key === 'a' || e.key === 'A') $('auto').click();
  else if (e.key === 'm' || e.key === 'M') $('manual').click();
  else if (e.key === 'r' || e.key === 'R') $('reset').click();
});

function colourOf(b, autoMode) {
  if (b.chosen) return COLORS.chosen;
  if (!b.choosable) return COLORS.off;
  if (b.status === 'end') return COLORS.end;
  return (autoMode && b.auto) ? COLORS.auto : COLORS.choosable;
}
function why(b) {
  if (b.status === 'direction') return 'not vertical' + (b.angle != null ? ' · ' + b.angle + '°' : '');
  if (b.status === 'end') return b.choosable ? 'too close to its end · choose to override' : 'too close to its end';
  return b.auto ? 'automatic pick' : 'can be chosen';
}
function renderBranches(s) {
  const list = $('branches'), bs = s.candidates || [], autoMode = !s.selection;
  if (!bs.length) {
    const li = document.createElement('li'); li.className = 'empty';
    li.textContent = (s.tstate === 'tracking' && s.point) ? 'locked — shown again while searching' : 'no branch detected';
    list.replaceChildren(li); return;
  }
  list.replaceChildren(...bs.map(b => {
    const li = document.createElement('li'), btn = document.createElement('button');
    btn.className = 'branch' + (b.stale ? ' stale' : '') + (b.chosen ? ' chosen' : '');
    btn.disabled = !b.choosable;
    btn.title = b.choosable ? 'Lock on branch ' + b.n : 'Can not be chosen: ' + why(b);
    const num = document.createElement('span'); num.className = 'num'; num.style.background = colourOf(b, autoMode); num.textContent = b.n;
    const mid = document.createElement('span');
    mid.innerHTML = '<div>Branch ' + b.n + (b.stale ? ' <span class="why">(seen just before)</span>' : '') + '</div><div class="why reason"></div>';
    mid.querySelector('.reason').textContent = why(b);
    const conf = document.createElement('span'); conf.className = 'conf'; conf.textContent = b.confidence.toFixed(2);
    btn.append(num, mid, conf);
    btn.onclick = () => { post('/api/select', {n: b.n}); toast('Branch ' + b.n + ' chosen'); };
    li.append(btn); return li;
  }));
}
async function poll() {
  try {
    const s = await (await fetch('/api/state', {cache:'no-store'})).json();
    $('conn').classList.add('on');
    const tracking = s.tstate === 'tracking' && s.point;
    const mode = tracking ? 'tracking' : (s.tstate === 'lost' ? 'lost' : 'searching');
    const pm = $('p-mode'); pm.textContent = mode.toUpperCase(); pm.className = 'pill state-' + mode;
    $('p-ctrl').textContent = s.ctrl ? s.ctrl : 'controller —';
    $('p-size').textContent = s.size != null ? 'size ' + s.size.toFixed(1) + ' px' : 'size —';
    $('p-rate').textContent = s.rate ? Math.round(s.rate) + ' Hz' : '— Hz';
    $('p-view').textContent = 'view ' + Math.round(s.view_fps) + ' fps';
    $('auto').classList.toggle('on', !s.manual);
    $('manual').classList.toggle('on', !!s.manual);
    renderBranches(s);
    const n = s.selection && s.selection.n;
    if (tracking && !last.tracking) toast(n ? 'Locked on branch ' + n : 'Locked');
    else if (!tracking && last.tracking) toast(s.selection && s.selection.waiting ? 'Manual: choose a branch' : 'Searching');
    last = {tracking};
  } catch (e) { $('conn').classList.remove('on'); }
  setTimeout(poll, 300);
}
poll();
</script></body></html>
"""
