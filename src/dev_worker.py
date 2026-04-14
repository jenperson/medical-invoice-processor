"""Development worker supervisor with automatic restart on source changes.

This script runs a child worker process (``discover.py``) and watches the ``src/``
tree for Python file changes. It is intended for local development so workflow
code edits are picked up without manually stopping and restarting the worker.

How it works:
- Starts a child process using the current Python interpreter.
- Recursively watches ``src/`` for ``*.py`` file events.
- Debounces rapid file system events to avoid restart storms.
- Restarts the child worker when a relevant change is detected.
- If the child worker crashes/exits, waits for the next file change and then
    starts it again.
- Handles ``Ctrl+C`` cleanly by terminating the child and stopping the watcher.

Key tuning knob:
- ``DEBOUNCE_SECONDS`` controls minimum time between restart triggers.
"""

import os
import signal
import subprocess
import sys
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

WATCH_DIR = os.path.join(os.path.dirname(__file__))
DEBOUNCE_SECONDS = 1.0


class _RestartHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self._last_trigger = 0.0
        self._needs_restart = False

    def on_any_event(self, event) -> None:  # type: ignore[override]
        if not event.src_path.endswith(".py"):
            return
        now = time.monotonic()
        if now - self._last_trigger < DEBOUNCE_SECONDS:
            return
        self._last_trigger = now
        self._needs_restart = True

    def consume_restart(self) -> bool:
        if self._needs_restart:
            self._needs_restart = False
            return True
        return False


def _start_worker() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "discover.py")],
    )


def _stop_worker(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> None:
    handler = _RestartHandler()
    observer = Observer()
    observer.schedule(handler, WATCH_DIR, recursive=True)
    observer.start()

    proc = _start_worker()
    print(f"[dev] Worker started (pid {proc.pid}). Watching {WATCH_DIR} for changes...")

    try:
        while True:
            time.sleep(0.3)
            if handler.consume_restart():
                print("[dev] Change detected, restarting worker...")
                _stop_worker(proc)
                proc = _start_worker()
                print(f"[dev] Worker restarted (pid {proc.pid}).")
            # If the worker crashed, stop and wait for a file change
            if proc.poll() is not None:
                print(f"[dev] Worker exited (code {proc.returncode}). Waiting for file changes to restart...")
                while not handler.consume_restart():
                    time.sleep(0.3)
                proc = _start_worker()
                print(f"[dev] Worker restarted (pid {proc.pid}).")
    except KeyboardInterrupt:
        print("\n[dev] Shutting down...")
        _stop_worker(proc)
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
