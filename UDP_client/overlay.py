"""Overlay of the IBVS recording (main_record.py): what the UAV gets, and why.

Target reticle (colour = state), the width caliper and its 9 cross-sections (green: both edges
found, grey: not), the stored target spot (diamond), image centre and alignment error, a zoom
inset of the target, a status panel, size history with time-to-contact (computed like the UAV:
slope of ln(size) over 1 s), event notices (LOCKED, LOST, PUT BACK, ...), the run's timeline and
a date / mode / code-version stamp.

Nothing is drawn during the flight: main_record.py only writes frame_record() of every frame
(<run>_overlay.jsonl) next to the raw video, so the overlay costs the loop nothing. After the
flight, render_run() -- `python3 overlay.py [run]` or render_overlay.sh -- draws
<run>_ibvs_overlay.mp4 from the two.
"""
import json
import math
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # plain OpenCV text then
    Image = None

FONT_FILES = ["/usr/share/fonts/truetype/nunito-sans/NunitoSans-VariableFont_YTLC,opsz,wdth,wght.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

# palette, RGB
GREEN, AMBER, RED, CYAN, VIOLET = (61, 220, 151), (255, 181, 71), (255, 92, 92), (79, 195, 247), (180, 140, 255)
WHITE, MUTED, GRAY, PANEL = (245, 247, 250), (160, 174, 192), (140, 150, 165), (15, 23, 42)
PERCH = (255, 45, 45)    # the perching point
STATE_COLOUR = {"tracking": GREEN, "nosize": AMBER, "lost": RED, "search": MUTED}
STATE_LABEL = {"tracking": "TRACKING", "nosize": "NO SIZE", "lost": "LOST", "search": "SEARCHING"}
_STATE_CODE = {k: i for i, k in enumerate(STATE_COLOUR)}
_STATE_LUT = np.array([(c[2], c[1], c[0]) for c in STATE_COLOUR.values()], dtype=np.uint8)
EVENT_COLOUR = {"LOCKED": GREEN, "RE-FOUND": GREEN, "TARGET STORED": VIOLET, "VIEW ADDED": VIOLET,
                "PUT BACK": VIOLET, "LOST": RED, "GAVE UP": RED, "RESET": AMBER, "RE-LOCKED": AMBER}

SIZE_HISTORY_S = 3.0     # size graph span
TTC_WINDOW_S = 1.0       # as on the UAV (ttc_window_s)
TTC_MIN_SAMPLES = 10     # as on the UAV (ttc_min_samples)
EVENT_SHOW_S = 1.2       # how long an event notice stays
STORED_SHOW_S = 1.0      # the stored spot is drawn while its last sighting is this recent
INSET_PX = 300           # zoom inset size
SLOW_PANEL_EVERY = 3     # status panel, size graph and stamp are redrawn every this many frames


def _bgr(c):
    return c[2], c[1], c[0]


def _p(v):
    return int(round(float(v[0]))), int(round(float(v[1])))


def _git(root, *args):
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return ""


def code_version(repo_root):
    """'superproject abc1234 · ibvs def5678' (+ '*' for uncommitted changes), or 'unversioned'."""
    parts = []
    for name, path in (("", repo_root), ("ibvs ", Path(repo_root) / "ibvs")):
        h = _git(path, "rev-parse", "--short", "HEAD")
        if h:
            dirty = "*" if _git(path, "status", "--porcelain", "--untracked-files=no") else ""
            parts.append(f"{name}{h}{dirty}")
    return "  ·  ".join(parts) if parts else "unversioned"


def _r(v, nd=1):
    return None if v is None else [round(float(x), nd) for x in np.asarray(v, dtype=float).ravel()]


def frame_record(ctx, pipeline, packet, rate_hz, raw_written):
    """This frame's overlay values as a JSON-able dict (main loop; cheap, no drawing)."""
    tr, st = pipeline.tracker, pipeline.size_tracker
    tm = getattr(pipeline, "target_model", None)
    pt = ctx.estimated_point
    rec = dict(t=round(time.monotonic(), 4), wall=round(time.time(), 3),
               tf=None if ctx.t_frame is None else round(ctx.t_frame, 4), raw=raw_written,
               rate=None if rate_hz is None else round(rate_hz, 1), point=_r(pt), packet=packet,
               events=[list(e) for e in ctx.debug.get("events", [])],
               feat=None if ctx.extracted_features is None
               else np.round(np.asarray(ctx.extracted_features, dtype=float), 1).tolist(),
               cand=_r(ctx.point), tstate=None if tr is None else tr.state,
               stored_target=bool(tm is not None and tm.active), views=None if tm is None else len(tm.views))
    if st is not None and pt is not None and st.direction is not None:
        rec["branch"] = dict(width=round(float(st.width), 2), dir=_r(st.direction, 4),
                             axis=_r(st.axis_point if st.axis_point is not None else pt),
                             normal=_r(st.last_normal, 4), half=round(float(st.last_half), 1),
                             prof=[[_r(q["centre"]), None if q["edges"] is None else [_r(q["edges"][0]), _r(q["edges"][1])]]
                                   for q in st.last_profiles])
    off, seen = getattr(pipeline, "model_offset", None), getattr(pipeline, "model_seen_t", None)
    if pt is not None and off is not None and seen is not None and time.monotonic() - seen <= STORED_SHOW_S \
            and tr is not None and tr.point is not None:
        rec["stored"] = _r(np.asarray(tr.point, dtype=float) + np.asarray(off, dtype=float))
    return rec


class Overlay:
    """Draws the overlay video of a run from its records (render_run), after the flight."""

    def __init__(self, mode, version):
        self.mode, self.version = mode, version
        self._fonts = {}
        self._cache = {}             # slow panels: key -> (frame no., box, inverse alpha, premultiplied colour)
        self._strip = None           # the run's timeline, drawn once
        self._n = 0

    def snapshots(self, records):
        """(record, drawing input) per frame, with the history each frame needs (size graph, events)."""
        states, had = [], False
        for r in records:
            if r.get("point") is not None:
                states.append("tracking" if (r.get("packet") or {}).get("size") else "nosize")
                had = True
            else:
                states.append("lost" if had and (r.get("tstate") != "idle" or r.get("stored_target")) else "search")
                if states[-1] == "search":
                    had = False
        sizes, events, last_point = deque(), deque(), None
        for i, (r, state) in enumerate(zip(records, states)):
            now, ts = r["t"], r.get("tf") or r["t"]
            pt = None if r.get("point") is None else np.array(r["point"], dtype=float)
            if pt is not None:
                last_point = pt
            pk = r.get("packet")
            if pk and pk.get("size"):
                sizes.append((ts, float(pk["size"])))
            while sizes and ts - sizes[0][0] > SIZE_HISTORY_S:
                sizes.popleft()
            for name, detail in r.get("events", []):
                events.append((now, name, detail))
            while events and now - events[0][0] > EVENT_SHOW_S:
                events.popleft()
            s = dict(state=state, point=pt, last_point=last_point, packet=pk, rate=r.get("rate"), wall=r["wall"],
                     sizes=list(sizes), events=list(events), now=now, timeline=states, timeline_pos=i,
                     mode=self.mode, version=self.version,
                     features=None if r.get("feat") is None else np.array(r["feat"], dtype=float).reshape(-1, 2),
                     candidate=None if r.get("cand") is None else np.array(r["cand"], dtype=float),
                     nfeat=0 if r.get("feat") is None else len(r["feat"]), views=r.get("views"))
            br = r.get("branch")
            if br is not None:
                s["branch"] = dict(width=br["width"], direction=np.array(br["dir"]), axis=np.array(br["axis"]),
                                   normal=None if br["normal"] is None else np.array(br["normal"]), half=br["half"],
                                   profiles=[(np.array(c), None if e is None else (np.array(e[0]), np.array(e[1])))
                                             for c, e in br["prof"]])
            if r.get("stored") is not None:
                s["stored"] = np.array(r["stored"], dtype=float)
            yield r, s

    # ------------------------------------------------------------------ drawing

    def _font(self, size, weight):
        key = (size, weight)
        if key not in self._fonts:
            f = None
            for path in FONT_FILES:
                if Path(path).exists():
                    f = ImageFont.truetype(path, size)
                    try:
                        f.set_variation_by_axes([weight if a.get("name") in (b"Weight", "Weight") else a["default"]
                                                 for a in f.get_variation_axes()])
                    except Exception:
                        pass
                    break
            self._fonts[key] = f or ImageFont.load_default()
        return self._fonts[key]

    def _panel(self, frame, box, draw, key=None):
        """Translucent drawing in `box` (x0, y0, x1, y1) of the BGR frame: draw(ImageDraw, x0, y0).
        With `key`, the drawing is reused for SLOW_PANEL_EVERY frames (only blended again)."""
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0), min(int(box[2]), W), min(int(box[3]), H)
        if x1 <= x0 or y1 <= y0:
            return
        c = self._cache.get(key)
        if c is not None and self._n - c[0] < SLOW_PANEL_EVERY and c[1] == (x0, y0, x1, y1):
            roi = frame[y0:y1, x0:x1]
            roi[:] = (roi * c[2] + c[3]).astype(np.uint8)
            return
        # draw on a transparent layer, then blend it in with numpy (no PIL round trip of the frame)
        layer = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        draw(ImageDraw.Draw(layer), x0, y0)
        rgba = np.asarray(layer, dtype=np.float32)
        a = rgba[:, :, 3:4] / 255.0
        inv, pre = 1.0 - a, rgba[:, :, 2::-1] * a
        if key is not None:
            self._cache[key] = (self._n, (x0, y0, x1, y1), inv, pre)
        roi = frame[y0:y1, x0:x1]
        roi[:] = (roi * inv + pre).astype(np.uint8)

    def _pill(self, frame, x, y, text, colour=WHITE, size=13, weight=600, dot=None):
        if Image is None:
            cv2.putText(frame, text, (int(x), int(y) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _bgr(colour), 1, cv2.LINE_AA)
            return
        f = self._font(size, weight)
        tw = f.getlength(text)
        pad = 10 + (16 if dot else 0)
        def d(dr, x0, y0):
            dr.rounded_rectangle((x - x0, y - y0, x - x0 + tw + pad + 10, y - y0 + size + 13), (size + 13) // 2,
                                 fill=PANEL + (195,))
            if dot:
                dr.ellipse((x - x0 + 10, y - y0 + (size + 13) // 2 - 5, x - x0 + 20, y - y0 + (size + 13) // 2 + 5), fill=dot + (255,))
            dr.text((x - x0 + pad, y - y0 + 5), text, font=f, fill=colour + (255,))
        self._panel(frame, (x, y, x + tw + pad + 12, y + size + 14), d)

    def _measurement(self, img, s, T, scale, thick):
        """Cross-sections, caliper, reticle and stored spot; T maps frame -> canvas coordinates."""
        colour = STATE_COLOUR[s["state"]]
        br = s.get("branch")
        if br is not None:
            n, half = br["normal"], br["half"]
            for centre, edges in br["profiles"]:
                if edges is not None:
                    cv2.line(img, _p(T(edges[0])), _p(T(edges[1])), _bgr(GREEN), thick, cv2.LINE_AA)
                    for e in edges:
                        cv2.circle(img, _p(T(e)), max(2, thick + 1), _bgr(GREEN), -1, cv2.LINE_AA)
                elif n is not None:
                    a, b = centre - half * n, centre + half * n
                    for f0 in np.arange(0.0, 1.0, 0.2):
                        cv2.line(img, _p(T(a + f0 * (b - a))), _p(T(a + (f0 + 0.1) * (b - a))), _bgr(GRAY), thick, cv2.LINE_AA)
            if s["state"] == "tracking":
                c, u, w = br["axis"], br["direction"], br["width"]
                nn = np.array([-u[1], u[0]])
                a, b = c - 0.5 * w * nn, c + 0.5 * w * nn
                cv2.line(img, _p(T(a - 3 * nn)), _p(T(b + 3 * nn)), _bgr(WHITE), thick + 1, cv2.LINE_AA)
                for e in (a, b):
                    cv2.line(img, _p(T(e - 7 * u)), _p(T(e + 7 * u)), _bgr(WHITE), thick + 1, cv2.LINE_AA)
        c = s["point"] if s["point"] is not None else s["last_point"]
        if c is None:
            return
        w = (s.get("branch") or {}).get("width") or ((s["packet"] or {}).get("size")) or 12.0
        r, k = max(2.4 * w, 30.0), 12.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                q = c + (sx * r, sy * r)
                if s["state"] == "lost":   # dashed corners at the last known point
                    cv2.line(img, _p(T(q)), _p(T(q - (sx * k / 2, 0))), _bgr(colour), thick + 2, cv2.LINE_AA)
                    cv2.line(img, _p(T(q)), _p(T(q - (0, sy * k / 2))), _bgr(colour), thick + 2, cv2.LINE_AA)
                else:
                    cv2.line(img, _p(T(q)), _p(T(q - (sx * k, 0))), _bgr(colour), thick + 2, cv2.LINE_AA)
                    cv2.line(img, _p(T(q)), _p(T(q - (0, sy * k))), _bgr(colour), thick + 2, cv2.LINE_AA)
        if s.get("stored") is not None:
            sd, dm = T(s["stored"]), 5.0 * math.sqrt(scale)
            cv2.polylines(img, [np.int32([sd + (0, -dm), sd + (dm, 0), sd + (0, dm), sd + (-dm, 0)])], True,
                          _bgr(VIOLET), thick + 1, cv2.LINE_AA)
        if s["point"] is not None:
            # the perching point (the px/py sent): red, with a white rim so it shows on the dark branch
            cv2.circle(img, _p(T(c)), 4 + 2 * thick, _bgr(WHITE), -1, cv2.LINE_AA)
            cv2.circle(img, _p(T(c)), 3 + 2 * thick, _bgr(PERCH), -1, cv2.LINE_AA)

    def render(self, frame, s):
        """Draw the overlay of snapshot `s` on `frame` (BGR, modified and returned)."""
        self._n += 1
        H, W = frame.shape[:2]
        raw = frame.copy()
        ident = lambda v: np.asarray(v, dtype=float)

        # features: small cyan dots (and the detection candidate while searching)
        if s["features"] is not None:
            for f in s["features"]:
                cv2.circle(frame, _p(f), 3, (140, 110, 40), -1, cv2.LINE_AA)
                cv2.circle(frame, _p(f), 2, _bgr(CYAN), -1, cv2.LINE_AA)
        if s["point"] is None and s["candidate"] is not None:
            cv2.circle(frame, _p(s["candidate"]), 14, _bgr(CYAN), 2, cv2.LINE_AA)
        self._measurement(frame, s, ident, 1.0, 1)

        # image centre + dashed alignment line
        ctr = np.array([W / 2.0, H / 2.0])
        cv2.circle(frame, _p(ctr), 7, _bgr(WHITE), 1, cv2.LINE_AA)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.line(frame, _p(ctr + (9 * dx, 9 * dy)), _p(ctr + (14 * dx, 14 * dy)), _bgr(WHITE), 1, cv2.LINE_AA)
        if s["point"] is not None:
            vec = s["point"] - ctr
            for t0 in np.arange(0.08, 0.9, 0.06):
                cv2.line(frame, _p(ctr + t0 * vec), _p(ctr + (t0 + 0.03) * vec), _bgr(WHITE), 1, cv2.LINE_AA)

        # zoom inset of the target
        c = s["point"] if s["point"] is not None else s["last_point"]
        IX, IY = W - INSET_PX - 24, 62
        if c is not None:
            w = (s.get("branch") or {}).get("width") or ((s["packet"] or {}).get("size")) or 20.0
            S = int(np.clip(3.7 * w, 60, 220))
            zx0, zy0 = int(np.clip(c[0] - S / 2, 0, W - S)), int(np.clip(c[1] - S / 2, 0, H - S))
            zoom = cv2.resize(raw[zy0:zy0 + S, zx0:zx0 + S], (INSET_PX, INSET_PX), interpolation=cv2.INTER_CUBIC)
            sc = INSET_PX / S
            self._measurement(zoom, s, lambda v: (np.asarray(v, dtype=float) - (zx0, zy0)) * sc, sc, 2)
            frame[IY:IY + INSET_PX, IX:IX + INSET_PX] = zoom
            cv2.rectangle(frame, (zx0, zy0), (zx0 + S, zy0 + S), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (IX - 2, IY - 2), (IX + INSET_PX + 1, IY + INSET_PX + 1), _bgr(WHITE), 2, cv2.LINE_AA)

        # the whole run's timeline (bottom), with where this frame is
        Wt = W - 340
        tl = s["timeline"]
        if tl:
            if self._strip is None or self._strip[0] is not tl:
                idx = (np.arange(Wt) * len(tl) / Wt).astype(int)
                codes = np.fromiter((_STATE_CODE[tl[i]] for i in idx), dtype=np.uint8, count=Wt)
                self._strip = (tl, _STATE_LUT[codes][None, :, :].repeat(6, axis=0).astype(np.float32))
            frame[H - 24:H - 18, 20:20 + Wt] = (0.8 * self._strip[1] + 0.2 * frame[H - 24:H - 18, 20:20 + Wt]).astype(np.uint8)
            x = int(20 + Wt * s["timeline_pos"] / len(tl))
            cv2.rectangle(frame, (x - 1, H - 29), (x + 1, H - 13), _bgr(WHITE), -1)

        if Image is None:
            cv2.putText(frame, STATE_LABEL[s["state"]], (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _bgr(STATE_COLOUR[s["state"]]), 2, cv2.LINE_AA)
            return frame
        self._text(frame, s, IX, c)
        return frame

    def _text(self, frame, s, IX, c):
        H, W = frame.shape[:2]
        colour = STATE_COLOUR[s["state"]]
        pk = s["packet"] or {}
        br = s.get("branch")

        # status panel
        rows = [("target", f"{int(s['point'][0])}, {int(s['point'][1])} px" if s["point"] is not None else "–"),
                ("size", f"{pk['size']:.1f} px" if pk.get("size") else "–"),
                ("age", f"{1000 * pk['age']:.0f} ms" if pk.get("age") is not None else "–"),
                ("rate", f"{s['rate']:.0f} Hz" if s["rate"] else "–"),
                ("features", f"{s['nfeat']}" + (f"  ·  views {s['views']}" if s["views"] is not None else ""))]
        f_chip, f_lab, f_val = self._font(17, 800), self._font(15, 500), self._font(17, 700)
        def hud(dr, x0, y0):
            dr.rounded_rectangle((0, 0, 250, 176), 14, fill=PANEL + (165,))
            dr.ellipse((16, 18, 28, 30), fill=colour + (255,))
            dr.text((36, 13), STATE_LABEL[s["state"]], font=f_chip, fill=colour + (255,))
            dr.text((150, 15), s["mode"], font=f_lab, fill=MUTED + (255,))
            for i, (lab, val) in enumerate(rows):
                dr.text((16, 46 + 25 * i), lab, font=f_lab, fill=MUTED + (255,))
                dr.text((102, 44 + 25 * i), val, font=f_val, fill=WHITE + (255,))
        self._panel(frame, (20, 20, 271, 197), hud, key=("hud", s["state"]))

        # labels at the target
        if br is not None and s["state"] == "tracking":
            u = br["direction"]
            b = br["axis"] + 0.5 * br["width"] * np.array([-u[1], u[0]])
            self._pill(frame, b[0] + 22, b[1] - 13, f"{br['width']:.1f} px")
        if s.get("stored") is not None:
            self._pill(frame, s["stored"][0] - 110, s["stored"][1] - 10, "stored spot", VIOLET, 12, 600)
        vec = None if s["point"] is None else s["point"] - np.array([W / 2.0, H / 2.0])
        if vec is not None and np.hypot(*vec) > 120:   # near the centre it would cover the width label
            m = np.array([W / 2.0, H / 2.0]) + 0.3 * vec
            self._pill(frame, m[0] + 10, m[1] - 30, f"Δ {vec[0]:+.0f}, {vec[1]:+.0f}", WHITE, 12, 600)

        # inset label, version stamp
        if c is not None:
            ok = sum(1 for _, e in (br or {}).get("profiles", []) if e is not None)
            lab = f"zoom   edges {ok}/{len(br['profiles'])}" if br and br["profiles"] else "zoom"
            self._pill(frame, IX + 10, 72, lab)
        stamp = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime(s["wall"])) + f"   ·   {s['mode']}   ·   {s['version']}"
        f_tiny = self._font(12, 500)
        sw = f_tiny.getlength(stamp)
        self._panel(frame, (W - 30 - sw, 18, W - 18, 36),
                    lambda dr, x0, y0: dr.text((W - 24 - sw - x0, 22 - y0), stamp, font=f_tiny, fill=MUTED + (255,)),
                    key="stamp")

        # newest event notice, fading out
        if s["events"]:
            t, name, detail = s["events"][-1]
            alpha = max(0.0, 1.0 - (s["now"] - t) / EVENT_SHOW_S)
            if alpha > 0.05:
                f_ev, f_det = self._font(17, 800), self._font(13, 600)
                tw = f_ev.getlength(name) + (f_det.getlength(detail) + 14 if detail else 0) + 58
                x = W // 2 - tw / 2
                ec = EVENT_COLOUR.get(name, WHITE)
                def ev(dr, x0, y0):
                    dr.rounded_rectangle((x - x0, 22 - y0, x - x0 + tw, 60 - y0), 19, fill=PANEL + (int(200 * alpha),))
                    dr.ellipse((x - x0 + 16, 35 - y0, x - x0 + 28, 47 - y0), fill=ec + (int(255 * alpha),))
                    dr.text((x - x0 + 38, 30 - y0), name, font=f_ev, fill=WHITE + (int(255 * alpha),))
                    if detail:
                        dr.text((x - x0 + 38 + f_ev.getlength(name) + 12, 33 - y0), detail, font=f_det, fill=MUTED + (int(255 * alpha),))
                self._panel(frame, (x, 22, x + tw + 1, 61), ev)

        # size history + time-to-contact
        sizes = s["sizes"]
        X0, Y0, Wp, Hp = W - 300, H - 170, 280, 132
        ttc = None
        if sizes:
            tnow = sizes[-1][0]
            rec = [(t, math.log(v)) for t, v in sizes if tnow - t <= TTC_WINDOW_S]
            if len(rec) >= TTC_MIN_SAMPLES:
                tm = sum(t for t, _ in rec) / len(rec); lm = sum(l for _, l in rec) / len(rec)
                stt = sum((t - tm) ** 2 for t, _ in rec)
                slope = sum((t - tm) * (l - lm) for t, l in rec) / stt if stt > 1e-9 else 0.0
                ttc = 1.0 / slope if slope > 1e-3 else None
        f_small, f_big = self._font(13, 600), self._font(26, 800)
        def graph(dr, x0, y0):
            dr.rounded_rectangle((0, 0, Wp, Hp), 14, fill=PANEL + (165,))
            dr.text((16, 10), f"SIZE · last {SIZE_HISTORY_S:.0f} s", font=f_small, fill=MUTED + (255,))
            dr.text((16, 28), f"TTC {ttc:.1f} s" if ttc else "TTC –", font=f_big, fill=WHITE + (255,))
            if len(sizes) >= 2:
                t0, t1 = sizes[0][0], sizes[-1][0]
                ls = [math.log(v) for _, v in sizes]
                lo, hi = min(ls), max(ls)
                if hi - lo < 0.16:
                    mid = 0.5 * (lo + hi); lo, hi = mid - 0.08, mid + 0.08
                pts = [(16 + (Wp - 32) * (t - t0) / max(t1 - t0, 1e-6), Hp - 16 - (Hp - 76) * (l - lo) / max(hi - lo, 1e-6))
                       for (t, _), l in zip(sizes, ls)]
                dr.line(pts, fill=GREEN + (255,), width=3, joint="curve")
                dr.ellipse((pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4), fill=GREEN + (255,))
                rng = f"{sizes[0][1]:.0f} – {sizes[-1][1]:.0f} px"
                dr.text((Wp - 16 - f_small.getlength(rng), 32), rng, font=f_small, fill=MUTED + (255,))
        self._panel(frame, (X0, Y0, X0 + Wp + 1, Y0 + Hp + 1), graph, key="graph")


def render_run(prefix):
    """Draw <prefix>_ibvs_overlay.mp4 from <prefix>_raw.mp4 and <prefix>_overlay.jsonl. Returns frames written."""
    prefix = str(prefix)
    lines = [json.loads(l) for l in open(prefix + "_overlay.jsonl") if l.strip()]
    head = lines[0] if lines and lines[0].get("header") else {}
    records = [r for r in lines if not r.get("header") and r.get("raw", True)]   # frames the raw video has
    cap = cv2.VideoCapture(prefix + "_raw.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    ov = Overlay(head.get("mode", "?"), head.get("version", "unversioned"))
    writer, n = None, 0
    for r, s in ov.snapshots(records):
        ok, frame = cap.read()
        if not ok:
            break
        out = ov.render(frame, s)
        if writer is None:
            writer = cv2.VideoWriter(prefix + "_ibvs_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (out.shape[1], out.shape[0]))
        writer.write(out)
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(records)} frames")
    cap.release()
    if writer is not None:
        writer.release()
    return n


if __name__ == "__main__":
    import sys
    rec_dir = Path(__file__).resolve().parent / "recordings"
    if len(sys.argv) > 1:
        runs = [a[:-len("_overlay.jsonl")] if a.endswith("_overlay.jsonl") else a.rstrip("_") for a in sys.argv[1:]]
    else:   # the newest run that has overlay data
        data = sorted(rec_dir.glob("*_overlay.jsonl"))
        if not data:
            sys.exit(f"No *_overlay.jsonl in {rec_dir} (record with main_record.py first)")
        runs = [str(data[-1])[:-len("_overlay.jsonl")]]
    for run in runs:
        print(f"Drawing the overlay of {run} ...")
        t0 = time.monotonic()
        n = render_run(run)
        print(f"Saved {n} frames to {run}_ibvs_overlay.mp4 ({time.monotonic() - t0:.0f} s)")
