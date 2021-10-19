#!/usr/bin/env python

import sys, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from mmtools import trackj, lifetime

# default values
input_filename = None
time_scale = 1.0
fitting_start = 0
output_folder = 'analyze'
suffix_tsv = ".txt"
suffix_graph = ".png"
stem_regression = '_regression'
stem_lifetime = '_lifetime'
stem_newbinding = '_newbinding'
stem_cumulative = '_cumulative'

# parse arguments
parser = argparse.ArgumentParser(description='Calculate lifetime and regression curves', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-x', '--time-scale', type = float, default=time_scale, \
                    help='interval of time-lapse (in seconds)')

parser.add_argument('-s', '--fitting-start', type = int, default=fitting_start, \
                    help='starting point of fitting')

parser.add_argument('input_file', default=input_filename, \
                    help='input TSV file or TrackJ CSV files')
args = parser.parse_args()

# set arguments
time_scale = args.time_scale
input_filename = args.input_file
fitting_start = args.fitting_start
output_stem = Path(input_filename).stem
output_stem = re.sub('\.ome$', '', output_stem, flags = re.IGNORECASE)
output_stem = re.sub('\_speckles$', '', output_stem, flags = re.IGNORECASE)

# prepare a folder
print("Making an output folder {0}.".format(output_folder))
Path(output_folder).mkdir(parents = True, exist_ok = True)

def output_path (stem_suffix, file_suffix):
    return Path(output_folder).joinpath(output_stem + stem_suffix + file_suffix)

def fitting_text (fitting, start = 0):
    text = "Off-rate = {0:.3f} /sec, Half-life = {1:.3f} sec (t >= {2})".format(fitting['koff'], fitting['halflife'], start)
    return text

# read TSV or TrackJ CSV file
if Path(input_filename).suffix.lower() == ".txt":
    print("Read TSV from {0}.".format(input_filename))
    spot_table = pd.read_csv(input_filename, comment = '#', sep = '\t')
elif Path(input_filename).suffix.lower() == ".csv":
    print("Read TrackJ CSV from {0}.".format(input_filename))
    spot_table = trackj.read_spots(input_filename)
else:
    raise Exception("Unknown file format.")

print("Total {0} spots and {1} trackings.".format(len(spot_table.total_index), len(spot_table.total_index.unique())))

# regression table
output_table = lifetime.regression(spot_table, time_scale = time_scale)
fitting = lifetime.fit_one_phase_decay(output_table.time, output_table.spotcount)
fitting_func = fitting['func']

# regression tsv
output_filename = output_path(stem_regression, suffix_tsv)
print("Output regression table to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep = '\t', index = False)

# regression graph
output_filename = output_path(stem_regression, suffix_graph)
print("Output regression graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Regression from t = 0", size = 'xx-large')
axes.plot(output_table.time, fitting_func(output_table.time), color = 'black', linestyle = ':')
axes.scatter(output_table.time, output_table.spotcount, color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            fitting_text(fitting), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)
print(".")

# lifetime table
output_table = lifetime.lifetime(spot_table, time_scale = time_scale)
fitting = lifetime.fit_one_phase_decay(output_table.time, output_table.spotcount, start = fitting_start)
fitting_func = fitting['func']

# lifetime tsv
output_filename = output_path(stem_lifetime, suffix_tsv)
print("Output lifetime table to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep = '\t', index = False)

# lifetime graph
output_filename = output_path(stem_lifetime, suffix_graph)
print("Output lifetime graph to {0}.".format(output_filename))

figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Lifetime distribution ({0} spots)".format(output_table.spotcount.sum()), size = 'xx-large')
width = output_table.time[0]
axes.plot(output_table.time, fitting_func(output_table.time), color = 'black', linestyle = ':')
axes.bar(output_table.time, output_table.spotcount, width = -width/2, align = 'edge', color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            fitting_text(fitting, fitting_start), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)
print(".")

# cumulative lifetime table
output_table = lifetime.cumulative(spot_table, time_scale = time_scale)
fitting = lifetime.fit_one_phase_decay(output_table.time, output_table.spotcount, start = fitting_start)
fitting_func = fitting['func']

# cumulative lifetime tsv
output_filename = output_path(stem_cumulative, suffix_tsv)
print("Output cumulative lifetime table to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep = '\t', index = False)

# cumulative lifetime graph
output_filename = output_path(stem_cumulative, suffix_graph)
print("Output cumulative lifetime graph to {0}.".format(output_filename))

figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Cumulative Lifetime ({0} spots)".format(output_table.spotcount[0]), size = 'xx-large')
width = output_table.time[0]
axes.plot(output_table.time, fitting_func(output_table.time), color = 'black', linestyle = ':')
axes.bar(output_table.time, output_table.spotcount, width = -width/2, align = 'edge', color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            fitting_text(fitting, fitting_start), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)
print(".")

# new binding table
output_table = lifetime.new_bindings(spot_table, time_scale = time_scale)
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
