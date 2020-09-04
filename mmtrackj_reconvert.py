#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas, time

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
input_file = open(input_filename, 'r')
trackj_lines = input_file.readlines()
trackj_lines = [line.rstrip() for line in trackj_lines]
input_file.close()

# parse TrackJ lines
total_index = 0
spot_list = []
for line in trackj_lines:
    if line.startswith('#'):
        if line.startswith('#%stop'):
            total_index = total_index + 1
    else:
        items = line.split()
        x, y, plane = float(items[0]), float(items[1]), int(items[2])
        spot_list.append([total_index, plane - 1, x, y])

# convert to a dataframe
spot_table = pandas.DataFrame(data = spot_list, columns = ['total_index', 'plane', 'x', 'y'])

# output data
output_file = open(output_filename, 'w', newline='')
output_file.write('## Reconverted from %s by mmtrackj_unconvert at %s\n' % (input_filename, time.ctime()))
spot_table.to_csv(output_file, sep='\t', index=False)
print("Output lifetime to %s." % (output_filename))
