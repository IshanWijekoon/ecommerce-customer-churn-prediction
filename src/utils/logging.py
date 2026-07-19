"""Lightweight logging setup for scripts and notebooks."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    *,
    force: bool = False,
    stream: Optional[object] = None,
) -> None:
    """Configure the root logger once for the process.

    Args:
        level: Logging level name or numeric level.
        force: Reconfigure even if setup already ran.
        stream: Optional stream override (defaults to ``sys.stderr``).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str = "churn", level: int | str | None = None) -> logging.Logger:
    """Return a named logger, ensuring default setup has run.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Optional per-logger level override.

    Returns:
        Configured :class:`logging.Logger`.
    """
    setup_logging()
    logger = logging.getLogger(name)
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    return logger
