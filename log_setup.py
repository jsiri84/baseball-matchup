"""Shared per-run logging setup for every entry script in this repo.

Calling ``setup_logging("script_name")`` once at the top of a script:

* Picks a timestamp folder under ``logs/<YYYY-MM-DD_HH-MM-SS>/``. The
  timestamp is shared across an entire daily run -- ``daily.py`` exports
  the env var ``BASEBALL_BOT_LOG_TS`` before launching subprocesses, and
  any child script that finds that env var reuses it instead of minting a
  fresh one. This keeps fetch / matchup / roundup logs from a single
  daily run together in one folder.
* Opens ``logs/<ts>/<script_name>.log`` for append.
* Tees ``sys.stdout`` and ``sys.stderr`` so every existing ``print`` call
  shows up both on the console (unchanged) and inside the log file. The
  ``logging`` module is also wired up to the same file via a
  ``StreamHandler`` so callers can ``import logging; logging.info(...)``
  if they want structured records.

Returns the absolute path to the log file so callers can announce it.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
LOGS_ROOT = ROOT / "logs"
ENV_VAR = "BASEBALL_BOT_LOG_TS"

_already_configured: dict[str, Path] = {}


class _Tee:
    """File-like wrapper that mirrors writes to two streams.

    Used to point ``sys.stdout`` / ``sys.stderr`` at both the original
    console stream and the per-run log file, without touching any
    ``print`` call site.
    """

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
                s.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


def _resolve_timestamp() -> str:
    ts = os.environ.get(ENV_VAR)
    if ts:
        return ts
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.environ[ENV_VAR] = ts
    return ts


def setup_logging(script_name: str) -> Path:
    """Wire up tee'd stdout/stderr + a logging handler for this run.

    Idempotent per script_name within the same process.
    """
    if script_name in _already_configured:
        return _already_configured[script_name]

    ts = _resolve_timestamp()
    run_dir = LOGS_ROOT / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{script_name}.log"

    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    header = (
        f"\n{'=' * 72}\n"
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"start {script_name} pid={os.getpid()} "
        f"argv={sys.argv}\n"
        f"{'=' * 72}\n"
    )
    log_file.write(header)
    log_file.flush()

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    root = logging.getLogger()
    if not any(getattr(h, "_baseball_bot_log", False) for h in root.handlers):
        handler = logging.StreamHandler(log_file)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        handler._baseball_bot_log = True  # type: ignore[attr-defined]
        root.addHandler(handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)

    _already_configured[script_name] = log_path
    return log_path
