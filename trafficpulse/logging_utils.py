# -*- coding: utf-8 -*-
"""Configuration centralisee du logging pour TrafficPulse."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure le logging pour l'ensemble de l'application.

    Args:
        level: Niveau de log ("DEBUG", "INFO", "WARNING" ou "ERROR").
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Les librairies tierces sont tres verbeuses par defaut, on les limite
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
