#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas
from mmtools import trackj

# default values
input_filename = None
output_filename = None
filename_suffix = '_reconv.txt'

# parse arguments
parser = argparse.ArgumentParser(description='Reconvert a TrackerJ CSV file to a tanitracer file', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output TSV file ([basename]%s by default)' % (filename_suffix))
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input TrackJ CSV file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    stem = re.sub('\_speckles$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TrackJ CSV file
print("Read TrackJ CSV from %s." % (input_filename))
trackj_handler = trackj.TrackJ(input_filename)
spot_table = trackj_handler.spot_table

# output data
print("Output lifetime to %s." % (output_filename))
output_file = open(output_filename, 'w', newline='')
trackj_handler.output_header(output_file)
spot_table.to_csv(output_file, sep='\t', index=False)
output_file.close()

