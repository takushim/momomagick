#!/usr/bin/env python

import sys, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from mmtools import trackj, lifetime

# default values
input_filenames = None
time_scale = 1.0
fitting_start = 0
output_folder = 'analyze'
output_stem = None
suffix_tsv = ".txt"
suffix_graph = ".png"
stem_regression = '_regr'
stem_lifetime = '_life'
stem_newbinding = '_bind'
stem_cumulative = '_cuml'

# parse arguments
parser = argparse.ArgumentParser(description='Calculate lifetime and regression curves', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-O', '--output-folder', default = output_folder, \
                    help='folder to output results ({0} if not specified)'.format(output_folder))

parser.add_argument('-o', '--output-stem', default = output_stem, \
                    help='stem of output filenames (the first filename if not specified)')

parser.add_argument('-x', '--time-scale', type = float, default = time_scale, \
                    help='interval of time-lapse (in seconds)')

parser.add_argument('-s', '--fitting-start', type = int, default = fitting_start, \
                    help='starting point of fitting')

parser.add_argument('input_files', nargs = '+', default = input_filenames, \
                    help='input TSV file or TrackJ CSV files')
args = parser.parse_args()

# set arguments
time_scale = args.time_scale
input_filenames = args.input_files
fitting_start = args.fitting_start

# set the stem of output filenames
output_stem = args.output_stem
if output_stem is None:
    output_stem = Path(input_filenames[0]).stem
    output_stem = re.sub('\.ome$', '', output_stem, flags = re.IGNORECASE)
    output_stem = re.sub('\_speckles$', '', output_stem, flags = re.IGNORECASE)

# prepare a folder
output_folder = args.output_folder
print("Making an output folder {0}.".format(output_folder))
Path(output_folder).mkdir(parents = True, exist_ok = True)

# functions for 
def output_path (stem_suffix, file_suffix):
    return Path(output_folder).joinpath(output_stem + stem_suffix + file_suffix)

def fitting_text (fitting):
    koff = fitting['koff']
    halflife = fitting['halflife']
    start = fitting['start']
    text = "Off-rate = {0:.3f} /sec, Half-life = {1:.3f} sec (t >= {2})".format(koff, halflife, start)
    return text

def merge_results (names, tables):
    max_index = np.argmax([len(table.frame) for table in tables])
    frames = tables[max_index].frame.to_list()
    times = tables[max_index].time.to_list()
    count_list = []
    for table in tables:
        counts = [0] * len(frames)
        max_len = len(table.spotcount)
        counts[0:max_len] = table.spotcount.to_list()
        count_list.append(counts)
    return {'frame': frames, 'time': times, 'name_list': names, 'count_list': count_list}

def dict_to_spot_table (result):
    table_dict = {'Result_{0}'.format(i): result['count_list'][i] for i in range(len(result['count_list']))}
    table = pd.DataFrame(table_dict)
    return table

def dict_to_output_table (result):
    table = pd.DataFrame({'time': result['time'], 'frame': result['frame']}, columns = ['time', 'frame'])
    for index in range(len(result['count_list'])):
        table[result['name_list'][index]] = result['count_list'][index]
    return table

# read TSV or TrackJ CSV file
spot_tables = []
for input_filename in input_filenames:
    if Path(input_filename).suffix.lower() == ".txt":
        spot_table = pd.read_csv(input_filename, comment = '#', sep = '\t')
    elif Path(input_filename).suffix.lower() == ".csv":
        spot_table = trackj.read_spots(input_filename)
    else:
        raise Exception("Unknown file format.")
    
    total_records = len(spot_table.total_index)
    total_tracks = len(spot_table.total_index.unique())
    print("{0}: {1} records and {2} tracks.".format(input_filename, total_records, total_records))
    spot_tables.append(spot_table)

############################################################
# regression table
############################################################
# combine results
results = [lifetime.regression(table, time_scale = time_scale) for table in spot_tables]
result_dict = merge_results(input_filenames, results)

# fitting
spot_table = dict_to_spot_table(result_dict)
spot_sum = spot_table.sum(axis = 1)
times = np.array(result_dict['time'])
fitting = lifetime.fit_one_phase_decay(times, spot_sum)
fitting_func = fitting['func']

# regression tsv
output_filename = output_path(stem_regression, suffix_tsv)
output_table = dict_to_output_table(result_dict)
print("Output regression table to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep = '\t', index = False)

# regression graph
output_filename = output_path(stem_regression, suffix_graph)
print("Output regression graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Regression from t = 0", size = 'xx-large')
axes.plot(times, fitting_func(times), color = 'black', linestyle = ':')

width = times[0] / 2
labels = result_dict['name_list']
offset = np.zeros_like(times, dtype = float)
for index in range(len(result_dict['count_list'])):
    counts = np.array(result_dict['count_list'][index])
    axes.bar(times, counts, bottom = offset, width = width, label = labels[index])
    offset += np.array(counts)

axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.5, \
          fitting_text(fitting), size = 'large', ha = 'right', va = 'top')
axes.legend()

figure.savefig(output_filename, dpi = 300)
print(".")

############################################################
# lifetime table
############################################################
# combine results
results = [lifetime.lifetime(table, time_scale = time_scale) for table in spot_tables]
result_dict = merge_results(input_filenames, results)

# fitting
spot_table = dict_to_spot_table(result_dict)
spot_sum = spot_table.sum(axis = 1)
times = np.array(result_dict['time'])
fitting = lifetime.fit_one_phase_decay(times, spot_sum, start = fitting_start)
fitting_func = fitting['func']

# lifetime tsv
output_filename = output_path(stem_lifetime, suffix_tsv)
output_table = dict_to_output_table(result_dict)
print("Output lifetime table to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep = '\t', index = False)

# lifetime graph
output_filename = output_path(stem_lifetime, suffix_graph)
print("Output lifetime graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Lifetime ({0} spots)".format(spot_sum.sum()), size = 'xx-large')
axes.plot(times, fitting_func(times), color = 'black', linestyle = ':')

width = times[0] / 2
labels = result_dict['name_list']
offset = np.zeros_like(times, dtype = float)
for index in range(len(result_dict['count_list'])):
    counts = np.array(result_dict['count_list'][index])
    axes.bar(times, counts, bottom = offset, width = width, label = labels[index])
    offset += np.array(counts)

axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.5, \
          fitting_text(fitting), size = 'large', ha = 'right', va = 'top')
axes.legend()

figure.savefig(output_filename, dpi = 300)
print(".")

############################################################
# cumulative lifetime table
############################################################
# combine results
results = [lifetime.cumulative(table, time_scale = time_scale) for table in spot_tables]
result_dict = merge_results(input_filenames, results)

# fitting
spot_table = dict_to_spot_table(result_dict)
spot_sum = spot_table.sum(axis = 1)
times = np.array(result_dict['time'])
fitting = lifetime.fit_one_phase_decay(times, spot_sum)
fitting_func = fitting['func']

# lifetime tsv
output_filename = output_path(stem_cumulative, suffix_tsv)
output_table = dict_to_output_table(result_dict)
print("Output cumulative lifetime table to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep = '\t', index = False)

# lifetime graph
output_filename = output_path(stem_cumulative, suffix_graph)
print("Output cumulative lifetime graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Cumulative lifetime ({0} spots)".format(spot_sum[0], size = 'xx-large'))
axes.plot(times, fitting_func(times), color = 'black', linestyle = ':')

width = times[0] / 2
labels = result_dict['name_list']
offset = np.zeros_like(times, dtype = float)
for index in range(len(result_dict['count_list'])):
    counts = np.array(result_dict['count_list'][index])
    axes.bar(times, counts, bottom = offset, width = width, label = labels[index])
    offset += np.array(counts)

axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.5, \
          fitting_text(fitting), size = 'large', ha = 'right', va = 'top')
axes.legend()

figure.savefig(output_filename, dpi = 300)
print(".")

############################################################
# new binding table
############################################################
# combine results
results = [lifetime.new_bindings(table, time_scale = time_scale) for table in spot_tables]
output_table = pd.concat(results)
mean_lifetime = output_table[output_table.plane > 0].lifetime.mean()

# new binding tsv
output_filename = output_path(stem_newbinding, suffix_tsv)
print("Output new-binding table to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep = '\t', index = False)

# new binding graph
output_filename = output_path(stem_newbinding, suffix_graph)
print("Output new-binding graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Binding plane and lifetime", size = 'xx-large')
axes.axhline(mean_lifetime, color = 'black', linestyle = ':')
axes.scatter(output_table.plane, output_table.lifetime, color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            "Mean lifetime = {0:.3f} sec (plane > 0)".format(mean_lifetime), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)
print(".")
