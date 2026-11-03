"""One GPU job at a time: the exclusive lock every grader entry point takes.

Muse Spark's builder agents and Claude run these scripts on the same M1 Pro
GPU, and two runs at once distort each other's timings. A run that finds the
GPU busy says so and waits its turn, so runs queue and never overlap. The lock
file is in /tmp, so the main checkout and the Muse Spark worktree share it, and
the operating system releases it when the process ends, however it ends.
"""

import fcntl
import os
import sys

LOCK = "/tmp/mobilemoe-gpu.lock"
_held = []  # keeps the locked file open for the life of the process


def hold(job):
    """Take the GPU for this process, waiting while another run has it."""
    f = open(LOCK, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "another run"
        print(f"{job}: waiting for the GPU ({holder})", file=sys.stderr, flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()}: {job}\n")
    f.flush()
    _held.append(f)
