#!/usr/bin/env python

import argparse
import numpy as np
from mmtools import stack, log, particles

# default values
input_filename = None
output_filename = None
output_suffix = "_fil.json"
lifetime_range = [0, 0]

# parse arguments
parser = argparse.ArgumentParser(description='Filter particles with various conditions.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output JSON filename ([basename]_{0} if not specified)'.format(output_suffix))

parser.add_argument('-n', '--no-regression', action = 'store_true', \
                    help='remove tracks starting from the first frame')

parser.add_argument('-l', '--lifetime-range', nargs = 2, metavar = ('MIN', 'MAX'), default = lifetime_range,
                    help='range of lifetime. set max = 0 for no restriction.')

parser.add_argument('-f', '--first-only', action = 'store_true', \
                    help='keep the first spot in each tracking record')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='input JSON file of tracking data.')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filename = args.input_file
no_regression = args.no_regression
first_only = args.first_only
lifetime_range = args.lifetime_range
if lifetime_range[1] == 0:
    lifetime_range[1] = np.inf

output_filename = args.output_file
if output_filename is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)

# read a JSON file and parse it
logger.info("Loading tracking record: {0}.".format(input_filename))
json_data = particles.load_json(input_filename)
spot_list = particles.parse_tree(json_data['spot_list'])

# filter spots
if no_regression:
    reg_set = {spot['track'] for spot in spot_list if spot['time'] == 0}
    logger.info("Removing spots for regression analysis: {0}.".format(reg_set))
    spot_list = [spot for spot in spot_list if spot['track'] not in reg_set]

logger.info("Filtering tracks using a lifetime range: {0}.".format(lifetime_range))
track_set = set()
max_track = max([spot['track'] for spot in spot_list])
for index in range(max_track):
    track_spots = [spot for spot in spot_list if spot['track'] == index]
    if len(track_spots) < lifetime_range[0] or lifetime_range[1] < len(track_spots):
        continue
    track_set = track_set | {index}
spot_list = [spot for spot in spot_list if spot['track'] in track_set]

if first_only:
    logger.info("Keeping the first spot in each tracking record.")
    spot_list = [spot for spot in spot_list if spot['parent'] is None]

# output a JSON file
logger.info("Output a json file: {0}.".format(output_filename))
json_data['spot_list'] = spot_list
particles.save_json(output_filename, json_data)


