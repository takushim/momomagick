#!/usr/bin/env python

import logging
from pathlib import Path

def get_logger (filename, level = 'INFO'):
    logging.basicConfig(format = '%(asctime)s %(name)s %(levelname)s: %(message)s')
    logger = logging.getLogger(Path(filename).name)
    logger.setLevel(level)
    return logger
