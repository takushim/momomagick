#!/usr/bin/env python

import argparse, json
import pandas as pd
from pathlib import Path
from mmtools import stack, log, particles

# default values
input_filenames = None
output_filename = None
output_suffix = "_{0}.txt"
graph_filename = None
graph_suffix = "_{0}.png"
analysis = 'classify'
time_scale = 1.0

# parse arguments
parser = argparse.ArgumentParser(description='Replace labels of particles.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output filenames ([basename]_{0} if not specified)'.format(output_suffix.format('[analysis]')))

parser.add_argument('-g', '--graph-file', default = graph_filename, \
                    help='graph filenames ([basename]_{0} if not specified)'.format(graph_suffix.format('[analysis]')))

parser.add_argument('-x', '--time-scale', type = float, default = time_scale, \
                    help='interval of time-lapse (in seconds)')

log.add_argument(parser)

parser.add_argument('input_files', nargs = '+', default = input_filenames, \
                    help='input JSON file of tracking data. Results from multiple files are merged.')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filenames = args.input_files
time_scale = args.time_scale
output_suffix = output_suffix.format(analysis)
graph_suffix = graph_suffix.format(analysis)

output_filename = args.output_file
if output_filename is None:
    output_filename = stack.with_suffix(input_filenames[0], output_suffix)

graph_filename = args.graph_file
if graph_filename is None:
    graph_filename = stack.with_suffix(input_filenames[0], graph_suffix)

# read JSON files
spot_tables = []
plane_counts = []
for input_filename in input_filenames:
    suffix = Path(input_filename).suffix.lower()
    with open(input_filename, 'r') as f:
        json_data = json.load(f)
        spot_table = pd.DataFrame(particles.parse_tree(json_data['spot_list']))
        spot_table['plane'] = spot_table['time']
        spot_table['total_index'] = spot_table['track']
        plane_count = json_data['image_properties']['t_count']
    
    total_records = len(spot_table.total_index)
    total_tracks = len(spot_table.total_index.unique())
    logger.info("{0}: {1} records and {2} tracks in {3} frames.".format(input_filename, total_records, total_tracks, plane_count))
    spot_tables.append(spot_table)
    plane_counts.append(plane_count)

# generate tables
for spot_table in spot_tables:
    print(spot_table.groupby('label').size())
