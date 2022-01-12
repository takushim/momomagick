#!/usr/bin/env python

import logging
from pathlib import Path

def get_logger (filename, level = 'INFO'):
    logger = logging.getLogger(Path(filename).name)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(level)

    return logger
