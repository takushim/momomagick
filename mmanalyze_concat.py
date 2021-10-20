#!/usr/bin/env python

import sys, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from mmtools import trackj

# default values
input_filenames = None
lifetime_drop_first = False
output_filename = None
output_suffix = '_concat.txt'

# parse arguments
parser = argparse.ArgumentParser(description='Concatenate tracking records', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image filename. [basename]{0} by default'.format(output_suffix))

parser.add_argument('input_files', nargs = '*', default = input_filenames, \
                    help='input TSV file or TrackJ CSV files (*.txt and *.csv if not specified)')
args = parser.parse_args()

# set arguments
input_filenames = args.input_files
if len(input_filenames) == 0:
    input_filename = list(Path.glob('*.txt')) + list(Path.glob('*.csv'))

output_filename = args.output_file
if args.output_file is None:
    output_stem = Path(input_filenames[0]).stem
    output_stem = re.sub('\.ome$', '', output_stem, flags = re.IGNORECASE)
    output_stem = re.sub('\_speckles$', '', output_stem, flags = re.IGNORECASE)
    output_filename = output_stem + output_suffix

# read TSV or TrackJ CSV file
spot_table = pd.DataFrame()
for input_filename in input_filenames:
    if Path(input_filename).suffix.lower() == ".txt":
        print("Read TSV from {0}.".format(input_filename))
        work_table = pd.read_csv(input_filename, comment = '#', sep = '\t')
    elif Path(input_filename).suffix.lower() == ".csv":
        print("Read TrackJ CSV from {0}.".format(input_filename))
        work_table = trackj.read_spots(input_filename)
    else:
        raise Exception("Unknown file format.")
    
    if len(spot_table) > 0:
        offset = np.max(spot_table.total_index) + 1
        work_table['total_index'] = work_table['total_index'] + offset

    spot_table = pd.concat([spot_table, work_table])

print("Total {0} spots and {1} trackings.".format(len(spot_table.total_index), len(spot_table.total_index.unique())))
print("Output to:", output_filename)
spot_table.to_csv(output_filename, sep = '\t', index = False)
