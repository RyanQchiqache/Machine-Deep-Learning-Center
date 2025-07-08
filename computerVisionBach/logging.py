import logging
from colorlog import ColoredFormatter
from loguru import logger
import sys


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            fmt="%(log_color)s[%(levelname)s] %(message)s",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            }
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def setup_logger(name: str, level: str = "DEBUG") -> logger:
    # Set the log level for the logger (default is DEBUG)
    logger.remove()  # Remove the default handler (if any)

    # Add a new handler with a colored formatter
    logger.add(
        sys.stdout,
        format="<bold>{time:YYYY-MM-DD HH:mm:ss}</bold> | <level>{level}</level> | <cyan>{message}</cyan>",
        level=level,
        colorize=True
    )

    return logger