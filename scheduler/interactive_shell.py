# interactive_shell.py
"""
Pause-time object browser for the Slurm-sim simulator.

Usage inside Controller
-----------------------
    from interactive_shell import InteractiveShell
    ...
    def pause(self):
        shell = InteractiveShell(self)          # self == controller
        shell.cmdloop()

The shell understands:

    ls              list children of current object
    cd <name | #>   descend;  cd .. / up / /  navigate
    cat <name | #>  pretty-print value
    pwd             show path
    resume | exit   leave the shell, resume sim
    wait_history    domain-specific helpers …
"""

import cmd
import inspect
import pprint
import shlex
import threading
from typing import Any, List, Tuple, Dict

PP = pprint.PrettyPrinter(compact=True, width=100)


# --------------------------------------------------------------------------- #
# helpers – treat dict/list/object attributes uniformly
# --------------------------------------------------------------------------- #
def children(obj: Any) -> List[Tuple[str, Any]]:
    """
    Return (key, child) pairs for *obj* so the browser can iterate.
    """
    if isinstance(obj, dict):
        return list(obj.items())
    if isinstance(obj, (list, tuple, set)):
        return [(str(i), c) for i, c in enumerate(obj)]
    out = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if inspect.isroutine(val):
            continue
        out.append((name, val))
    return out


def is_leaf(obj: Any) -> bool:
    if isinstance(obj, (dict, list, tuple, set)):
        return len(obj) == 0
    return len(children(obj)) == 0


def describe(obj: Any) -> str:
    if isinstance(obj, dict):
        return f"dict len {len(obj)}"
    if isinstance(obj, list):
        return f"list len {len(obj)}"
    if isinstance(obj, tuple):
        return f"tuple len {len(obj)}"
    if isinstance(obj, set):
        return f"set len {len(obj)}"
    return type(obj).__name__


# --------------------------------------------------------------------------- #
# the shell
# --------------------------------------------------------------------------- #
class InteractiveShell(cmd.Cmd):
    intro = (
        "\n--- Simulation paused; browse objects with ls / cd / cat ------------"
        "\nType 'help' or '?' for command list, 'resume' to continue.\n"
    )
    prompt = "(inspect /) "

    # life-cycle
    def __init__(self, controller, extra_roots: Dict[str, Any] | None = None):
        """
        Parameters
        ----------
        controller : Controller
            The live simulator instance – exposed under key ``ctl``.
        extra_roots : dict[str, Any] | None
            Any other top-level objects you’d like to browse.
        """
        super().__init__()
        self.ctrl = controller
        self._roots = {
            "ctl": controller,
            "queue": controller.queue,
            "parts": controller.partitions,
            **(extra_roots or {}),
        }
        self._path: List[Tuple[str, Any]] = []  # stack of (name, obj)

    # shorthand for “current object”
    @property
    def cur(self) -> Any:
        return self._path[-1][1] if self._path else self._roots

    # nice pwd string
    def _pwd(self):
        return "/" + "/".join(p[0] for p in self._path) if self._path else "/"

    # update prompt after each command
    def postcmd(self, stop, line):
        self.prompt = f"(inspect {self._pwd()}) "
        return stop

    # basic nav
    def do_ls(self, _):
        "ls – list children"
        if is_leaf(self.cur):
            print(" <leaf value>", type(self.cur).__name__)
            return
        for i, (name, child) in enumerate(children(self.cur)):
            print(f"[{i:>3}] {name:<20} {describe(child)}")

    def do_cd(self, arg):
        "cd NAME|#  – descend;   cd .. | up | / to navigate"
        arg = arg.strip()
        if arg in ("..", "up"):
            if self._path:
                self._path.pop()
            return
        if arg in ("/", ""):
            self._path.clear()
            return

        # resolve
        kids = children(self.cur)
        target = None
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(kids):
                target = kids[idx]
        else:
            for k, v in kids:
                if k == arg:
                    target = (k, v)
                    break
        if target is None:
            print("no such child")
        else:
            self._path.append(target)

    # alias
    do_up = lambda self, arg: self.do_cd("..")

    def complete_cd(self, text, *_):
        names = [k for k, _ in children(self.cur)]
        return [n for n in names if n.startswith(text)]

    def do_pwd(self, _):
        "pwd – show path"
        print(self._pwd())

    def do_cat(self, arg):
        "cat [NAME|#] – pretty-print value (default: current object)"
        arg = arg.strip()
        obj = self.cur
        if arg:
            kids = children(self.cur)
            if arg.isdigit():
                idx = int(arg)
                if 0 <= idx < len(kids):
                    obj = kids[idx][1]
                else:
                    print("index out of range"); return
            else:
                for k, v in kids:
                    if k == arg:
                        obj = v
                        break
                else:
                    print("no such child"); return
        PP.pprint(obj)

    # sim helpers
    def do_wait_history(self, arg):
        """
        wait_history [-n COUNT] [partition]
            Show last 5 wait-reason entries for the first COUNT queued jobs.
        """
        toks = shlex.split(arg)
        count = 10
        part = None
        if toks[:2] == ["-n", *toks[1:2]]:
            count = int(toks[1]); toks = toks[2:]
        if toks:
            part = toks[0]

        jobs = [
            j for j in self.ctrl.queue.queue
            if part is None or j.partition.name == part
        ]
        if not jobs:
            print("No queued jobs match.")
            return

        for j in jobs[:count]:
            wait = (self.ctrl.time - j.submit) if j.submit else "?"
            print(f"{j.jid:<10} state={j.state.name:<10} wait={wait}")
            for ts, reason in j.wait_history[-5:]:
                print(f"   {ts}: {reason}")

    def do_last_skip(self, arg):
        "last_skip [partition] – print last skip reason for each queued job."
        part = arg.strip() or None
        for j in self.ctrl.queue.queue:
            if part and j.partition.name != part:
                continue
            print(f"{j.jid:<10} → {j.last_skip}")

    def do_stats(self, _):
        "stats – one-line cluster status (queued / running)"
        print(f"Time={self.ctrl.time}  "
              f"Running={len(self.ctrl.running_jobs)}  "
              f"Queued={len(self.ctrl.queue.queue)}")

    def do_step(self, arg):
        """
        step [N] – execute N internal _step() iterations then return.

        Requires Controller to expose a thread-safe `_step_lock`.
        """
        n = int(arg or 1)
        lock: threading.Lock | None = getattr(self.ctrl, "_step_lock", None)
        if lock is None:
            print("Controller has no _step_lock; step unavailable.")
            return
        for _ in range(n):
            self.ctrl.paused = False
            lock.acquire()
            try:
                self.ctrl._step(False, None, False, False)
            finally:
                lock.release()
            self.ctrl.paused = True
        print(f"Executed {n} manual step(s).")

    # resume / exit
    def do_resume(self, _):
        "resume – leave inspector and continue simulation"
        return True

    do_exit = do_quit = do_q = do_resume