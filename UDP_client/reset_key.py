"""Keys in the terminal: 'r' drops the current target (and the live view's branch choice) and searches
for a new point; 1-9 choose that
branch (numbers as in the live view), 'a' returns to the automatic branch choice, 'm' locks on nothing
until a branch is chosen."""
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
            elif ch in "123456789" or ch.lower() in "am":
                import live_view
                lv = live_view.current()
                if lv is None:
                    continue
                if ch.lower() == "a":
                    lv.command("auto")
                elif ch.lower() == "m":
                    lv.command("manual")
                else:
                    lv.command("number", int(ch))

    threading.Thread(target=loop, daemon=True).start()
    print("Press 'r' to reset the target and search again; 1-9 choose a branch, 'm' wait for a choice, 'a' automatic")


def check(pipeline, ctx):
    """Call once per frame: on 'r', forget the tracked/lost target (back to detection)."""
    if _pressed.is_set():
        _pressed.clear()
        ctx.estimated_point = None  # this frame's point belongs to the dropped target
        pipeline.reset_target()     # also forgets the stored target, so a new point gets chosen
        print("[reset] 'r' pressed — target dropped, searching for a new point")
        import live_view
        lv = live_view.current()
        if lv is not None:          # and the branch choice: in manual mode, choose again
            lv.clear_choice()
