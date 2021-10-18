#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas
from mmtools import trackj

# default values
input_filename = None
output_filename = None
filename_suffix = '_reconv.txt'
scaling = 1.0

# parse arguments
parser = argparse.ArgumentParser(description='Reconvert a TrackerJ CSV file to a tanitracer file', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output TSV file ([basename]%s by default)' % (filename_suffix))
parser.add_argument('-x', '--scaling', type = float, default = scaling, \
                    help='Scale cooredinates to use magnified images')
parser.add_argument('input_file', default = input_filename, \
                    help='input TrackJ CSV file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
scaling = args.scaling
if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    stem = re.sub('\_speckles$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file

# read TrackJ CSV file
print("Read TrackJ CSV from %s." % (input_filename))
spot_table = trackj.read_spots(input_filename)

# scale coodinates
print("Scaling factor:", scaling)
spot_table['x'] = spot_table['x'] * scaling
spot_table['y'] = spot_table['y'] * scaling

# output data
print("Output lifetime to %s." % (output_filename))
output_file = open(output_filename, 'w', newline='')
trackj.output_header(output_file, input_filename)
spot_table.to_csv(output_file, sep='\t', index=False)
output_file.close()

