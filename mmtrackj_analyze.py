#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas
from mmtools import trackj, lifetime

# default values
input_filename = None
time_scale = 1.0
filename_regression_suffix = '_regression.txt'
filename_lifetime_suffix = '_lifetime.txt'

# parse arguments
parser = argparse.ArgumentParser(description='Analyze a TrackerJ CSV file', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-x', '--time-scale', nargs = 1, type = float, \
                    metavar = ('SCALE'), default=[time_scale], \
                    help='interval of time-lapse (in seconds)')
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input TrackJ CSV file')
args = parser.parse_args()

# set arguments
time_scale = args.time_scale[0]
input_filename = args.input_file[0]
stem = pathlib.Path(input_filename).stem
stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
output_regression_filename = stem + filename_regression_suffix
output_lifetime_filename = stem + filename_lifetime_suffix

# read TrackJ CSV file
print("Read TrackJ CSV from %s." % (input_filename))
spot_table = trackj.TrackJ(input_filename).spot_table

# lifetime
lifetime_analyzer = lifetime.Lifetime(spot_table, time_scale)

# regression
output_table = lifetime_analyzer.regression()
output_table.to_csv(output_regression_filename, sep='\t', index=False)
print("Output regression to %s." % (output_regression_filename))

# lifetime
output_table = lifetime_analyzer.lifetime()
output_table.to_csv(output_lifetime_filename, sep='\t', index=False)
print("Output lifetime to %s." % (output_lifetime_filename))
