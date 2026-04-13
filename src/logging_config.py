import logging
import time

def silence_noisy_loggers():
    for name in ("mistralai_workflows", "httpx", "httpcore", "temporalio"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Workflow logger ───────────────────────────────────────────────────────────

_wf_count = 0
_started: set[str] = set()    # wf_ids that have printed start line
_completed: set[str] = set()  # wf_ids that have printed done line
_active: set[str] = set()     # wf_ids currently allowed to log
_batch_mode_wfs: dict[str, int] = {}  # wf_ids running in batch mode -> workflow number
_wf_start_times: dict[str, float] = {}  # wf_ids -> start time


def wf_start(wf_id: str, filename: str, is_batch_mode: bool = False) -> None:
    global _wf_count
    if wf_id in _completed:
        # workflow finished but Temporal is doing a closing replay — stay silent
        return
    _active.add(wf_id)
    if wf_id in _started:
        # mid-workflow replay (e.g. after signal) — restore active but don't reprint header
        return
    _wf_count += 1
    _started.add(wf_id)
    _wf_start_times[wf_id] = time.time()

    if is_batch_mode:
        _batch_mode_wfs[wf_id] = _wf_count
        print(f"workflow #{_wf_count} ({wf_id}) started")
    else:
        print(f"\n[{wf_id}] ┌─ workflow #{_wf_count}  {filename}")


def wf_activity(name: str, detail: str = "", wf_id: str = "") -> None:
    if wf_id not in _active or wf_id in _batch_mode_wfs:
        return

    # Full mode only: with anchors
    suffix = f"  {detail}" if detail else ""
    print(f"[{wf_id}] │  → {name}{suffix}")


def wf_waiting(wf_id: str, confidence: float, threshold: float) -> None:
    if wf_id not in _active or wf_id in _batch_mode_wfs:
        return

    print(f"[{wf_id}] │  ⏸  waiting for human  (confidence={confidence:.0%} < {threshold:.0%})")


def wf_signal(wf_id: str, category: str) -> None:
    if wf_id not in _active or wf_id in _batch_mode_wfs:
        return

    print(f"[{wf_id}] │  ✓  signal received  category={category}")


def wf_done(wf_id: str, category: str, confidence: float) -> None:
    if wf_id not in _active:
        return
    if wf_id in _completed:
        return
    _completed.add(wf_id)
    _active.discard(wf_id)

    # Calculate duration
    elapsed = time.time() - _wf_start_times.get(wf_id, time.time())
    duration_str = f"{elapsed:.1f}s"

    if wf_id in _batch_mode_wfs:
        wf_num = _batch_mode_wfs[wf_id]
        print(f"workflow #{wf_num} ({wf_id}) done - {category} {confidence:.0%} ({duration_str})")
    else:
        print(f"[{wf_id}] └─ done  category={category}  confidence={confidence:.0%}  ({duration_str})")
