from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False

_VALID_LEVELS = {
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
}


def _env_level(default: str = "INFO") -> str:
    """
    Read LOGGING_LEVEL from env. Treat empty string as unset.
    """
    raw = os.getenv("LOGGING_LEVEL")
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    return raw.upper()


def configure_logging(*, level: Optional[str] = None, force: bool = False) -> None:
    """
    Configure root logging once for this process.

    - Level comes from argument > LOGGING_LEVEL env var > default INFO
    - Logs to stdout (good for Docker/Airflow)
    - Avoids double configuration unless force=True
    """
    global _CONFIGURED

    if _CONFIGURED and not force:
        return

    lvl = (level.strip().upper() if level and level.strip() else _env_level("INFO"))

    if lvl not in _VALID_LEVELS:
        raise ValueError(f"Invalid LOGGING_LEVEL {lvl!r}. Valid: {sorted(_VALID_LEVELS)}")

    numeric_level = getattr(logging, lvl)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    # Airflow may preconfigure handlers; if so, don't add another unless force=True.
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)