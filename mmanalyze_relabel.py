#!/usr/bin/env python

import argparse, json
from numpyencoder import NumpyEncoder
from mmtools import stack, log

# default values
input_filename = None
output_filename = None
output_suffix = "_label.json"
label = None

# parse arguments
parser = argparse.ArgumentParser(description='Replace labels of particles.', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output filenames ([basename]_{0} if not specified)'.format(output_suffix))

parser.add_argument('-l', '--label', default = label, \
                    help='label to replace (None to remove)')

log.add_argument(parser)

parser.add_argument('input_file', default = input_filename, \
                    help='input JSON file of tracking data.')

args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
input_filename = args.input_file
label = args.label
output_filename = args.output_file
if output_filename is None:
    output_filename = stack.with_suffix(input_filename, output_suffix)

# read a JSON file
with open(input_filename, 'r') as f:
    json_data = json.load(f)

# replace labels
for spot in json_data['spot_list']:
    if spot['parent'] is None:
        logger.info("Spot #{0}: {1} -> {2}.".format(spot['index'], spot['label'], label))
        spot['label'] = label

# output a JSON file
with open(output_filename, 'w') as f:
    json.dump(json_data, f, ensure_ascii = False, indent = 4, sort_keys = False, \
              separators = (',', ': '), cls = NumpyEncoder)
