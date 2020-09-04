#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas

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
input_file = open(input_filename, 'r')
trackj_lines = input_file.readlines()
trackj_lines = [line.rstrip() for line in trackj_lines]
input_file.close()

# parse TrackJ lines
total_index = 0
life_count = 1
spot_list = []
for line in trackj_lines:
    if line.startswith('#'):
        if line.startswith('#%stop'):
            total_index = total_index + 1
            life_count = 1
    else:
        items = line.split()
        x, y, plane = float(items[0]), float(items[1]), int(items[2])
        spot_list.append([total_index, plane, x, y, life_count])
        life_count = life_count + 1

# convert to a dataframe
spot_table = pandas.DataFrame(data = spot_list, columns = ['total_index', 'plane', 'x', 'y', 'life_count'])

# regression
work_table = spot_table.copy()

# spots to be counted
index_set = set(work_table[work_table.plane == 1].total_index.tolist())
print("Regression set:", index_set)

# regression
output_indexes = []
output_counts = []
for index in range(0, work_table.plane.max() + 1):
    spot_count = len(work_table[(work_table.total_index.isin(index_set)) & (work_table.plane == (index + 1))])
    output_indexes += [index]
    output_counts += [spot_count]

# output data
output_columns = ['lifecount', 'lifetime', 'regression']
output_times = [i * time_scale for i in output_indexes]
output_table = pandas.DataFrame({ \
                    output_columns[0] : output_indexes, \
                    output_columns[1] : output_times, \
                    output_columns[2] : output_counts}, \
                    columns = output_columns)
output_table.to_csv(output_regression_filename, sep='\t', index=False)
print("Output regression to %s." % (output_regression_filename))

# lifetime
work_table = spot_table.drop_duplicates(subset='total_index', keep='last').reset_index(drop=True)

# prepare data
output_columns = ['lifecount', 'lifetime', 'spotcount']
lifecount_max = work_table.life_count.max()
output_indexes = [i for i in range(1, lifecount_max + 1)]
output_times = [i * time_scale for i in output_indexes]
output_counts = [len(work_table[work_table.life_count == i]) for i in output_indexes]

# output data
output_table = pandas.DataFrame({ \
                    output_columns[0] : output_indexes, \
                    output_columns[1] : output_times, \
                    output_columns[2] : output_counts}, \
                    columns = output_columns)
output_table.to_csv(output_lifetime_filename, sep='\t', index=False)
print("Output lifetime to %s." % (output_lifetime_filename))
