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

def add_argument (parser, short_option = '-L', long_option = '--log-level', default_level = 'INFO'):
    parser.add_argument('-L', '--log-level', default = default_level, \
                        help='Log level: DEBUG, INFO, WARNING, ERROR or CRITICAL')

