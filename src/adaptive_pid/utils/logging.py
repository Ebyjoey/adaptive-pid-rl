"""Structured logging configuration shared by training, evaluation, and ROS2 nodes.

A single ``get_logger`` entry point ensures consistent formatting (with
timestamps and module names) across the whole repo, and gives us one place
to redirect output to a file handler for a given experiment run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, *, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Idempotent: calling this multiple times for the same ``name`` will not
    add duplicate handlers (a common source of doubled log lines when a
    module is imported more than once, e.g. in tests).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if name in _CONFIGURED_LOGGERS:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED_LOGGERS.add(name)
    return logger
