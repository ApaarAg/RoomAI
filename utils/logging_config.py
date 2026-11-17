"""Logging configuration for RoomAI."""

import logging
import sys


def get_logger(name: str = "RoomAI") -> logging.Logger:
    """Configures and returns a structured logger for RoomAI."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False
    return logger
