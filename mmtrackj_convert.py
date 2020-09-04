#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas

# default values
input_filename = None
output_filename = None
filename_suffix = '_trackj.csv'
max_start_plane = 50
scaling = 1

# parse arguments
parser = argparse.ArgumentParser(description='Convert TaniTracer TSV file into TrackerJ CSV file', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output CSV file ([basename]%s by default)' % (filename_suffix))
parser.add_argument('-m', '--max-start-plane', nargs=1, default=[max_start_plane], \
                    help='Maximum plane number of tracking start')
parser.add_argument('-x', '--scaling', nargs=1, default=[scaling], \
                    help='Scale cooredinates to use magnified images')
parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input TSV file')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
max_start_plane = args.max_start_plane[0]
scaling = args.scaling[0]
if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TSV file
print("Read spots from %s." % (input_filename))
spot_table = pandas.read_csv(input_filename, comment = '#', sep = '\t')

# limit the span of tracking
print("Maxmum starting plane:", max_start_plane)
work_table = spot_table.drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)
index_set = set(work_table[work_table.plane < max_start_plane].total_index.tolist())
spot_table = spot_table[spot_table.total_index.isin(index_set)]

# scale coodinates
print("Scaling factor:", scaling)
spot_table['x'] = spot_table['x'] * float(scaling)
spot_table['y'] = spot_table['y'] * float(scaling)

# increment plane number for TrackerJ
spot_table['plane'] = spot_table['plane'] + 1

# open CSV file and output header
output_file = open(output_filename, 'w', newline='')
output_file.write("#speckles csv ver 1.2" + '\n')
output_file.write("#x(double)\ty(double)\tsize(double)\tframe(int)\ttype(int)" + '\n')

# loop for output
for index, spots in spot_table.groupby('total_index'):
    output_file.write("#%start speckle%" + '\n')
    spots.to_csv(output_file, columns = ['x', 'y', 'plane'], sep='\t', index=False, header=False, mode='a')
    output_file.write("#%stop speckle%" + '\n')

print("Output csv file to %s." % (output_filename))
