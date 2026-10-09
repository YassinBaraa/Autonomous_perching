"""Press 'r' in the terminal to drop the current target and search for a new point."""
import atexit
import sys
import termios
import threading
import tty

_pressed = threading.Event()


def start():
    if not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # single keys without Enter; Ctrl+C still works
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)

    def loop():
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                return
            if ch.lower() == "r":
                _pressed.set()

    threading.Thread(target=loop, daemon=True).start()
    print("Press 'r' to reset the target and search again")


def check(pipeline, ctx):
    """Call once per frame: on 'r', forget the tracked/lost target (back to detection)."""
    if _pressed.is_set():
        _pressed.clear()
        ctx.estimated_point = None  # this frame's point belongs to the dropped target
        pipeline.tracker.unlock()
        pipeline._lost_t0 = None
        print("[reset] 'r' pressed — target dropped, searching for a new point")
