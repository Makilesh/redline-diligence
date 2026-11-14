# src/utils/logger.py
# Structured JSON logging using standard library logging + python-json-logger
"""
Structured JSON logging configuration.
Uses standard library logging with python-json-logger for structured output.
All code uses logger.info(), logger.warning() — standard API.
"""

import logging
import os
import sys
from pythonjsonlogger import jsonlogger


def _configured_level() -> int:
    """The LOG_LEVEL env var as a logging level; INFO when unset or unrecognised."""
    name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, name, None)
    return level if isinstance(level, int) else logging.INFO


def setup_logger(
    name: str = "manda_rag",
    level: str | None = None,
) -> logging.Logger:
    """
    Creates and configures a structured JSON logger.

    LOG_LEVEL was documented in `.env.example` and read by nothing: every
    caller took the default and got INFO. It is now applied — to the handler,
    not the logger. The logger itself stays at INFO or lower so that records
    are still *created* at INFO: the metrics layer counts LLM calls from the
    wrapper's INFO usage record (src/utils/metrics.py), and that must not stop
    when someone turns the console down to WARNING.

    Args:
        name: Logger name (typically module name).
        level: Output level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults
            to the LOG_LEVEL environment variable, else INFO.

    Returns:
        Configured logger instance with JSON formatting.
    """
    logger = logging.getLogger(name)
    output_level = (
        getattr(logging, level.upper(), logging.INFO) if level else _configured_level()
    )

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(output_level)
    logger.setLevel(min(output_level, logging.INFO))
    return logger


# Module-level default logger
logger = setup_logger()
