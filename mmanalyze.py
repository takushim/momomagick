#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas
from matplotlib import pyplot
from mmtools import trackj, lifetime

# default values
input_filename = None
time_scale = 1.0
output_graphs = False
filename_regression_suffix = '_regression.txt'
filename_lifetime_suffix = '_lifetime.txt'
filename_newbinding_suffix = '_newbinding.txt'
filename_cumulative_suffix = '_cumulative.txt'
output_folder = 'analyze'

# parse arguments
parser = argparse.ArgumentParser(description='Calculate lifetime and regression curves', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-x', '--time-scale', type = float, default=time_scale, \
                    help='interval of time-lapse (in seconds)')

parser.add_argument('-G', '--output-graphs', action='store_true', default = output_graphs, \
                   help='output graphs for analyzed data')

parser.add_argument('input_file', default=input_filename, \
                    help='input TSV file or TrackJ CSV file')
args = parser.parse_args()

# set arguments
time_scale = args.time_scale
input_filename = args.input_file
stem = pathlib.Path(input_filename).stem
stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
stem = re.sub('\_speckles$', '', stem, flags=re.IGNORECASE)
output_graphs = args.output_graphs

# output files
output_regression_filename = pathlib.Path(output_folder).joinpath(stem + filename_regression_suffix)
output_lifetime_filename = pathlib.Path(output_folder).joinpath(stem + filename_lifetime_suffix)
output_newbinding_filename = pathlib.Path(output_folder).joinpath(stem + filename_newbinding_suffix)
output_cumulative_filename = pathlib.Path(output_folder).joinpath(stem + filename_cumulative_suffix)

# prepare a folder
print("Making an output folder {0}.".format(input_filename))
pathlib.Path(output_folder).mkdir(parents = True, exist_ok = True)

# read TSV or TrackJ CSV file
if pathlib.Path(input_filename).suffix.lower() == ".txt":
    print("Read TSV from {0}.".format(input_filename))
    spot_table = pandas.read_csv(input_filename, comment = '#', sep = '\t')
elif pathlib.Path(input_filename).suffix.lower() == ".csv":
    print("Read TrackJ CSV from {0}.".format(input_filename))
    spot_table = trackj.TrackJ(input_filename).spot_table
else:
    raise Exception("Unknown file format.")

# prepare an analyzer
print("Total {0} spots and {1} trackings.".format(len(spot_table.total_index), len(spot_table.total_index.unique())))
lifetime_analyzer = lifetime.Lifetime(spot_table, time_scale)

# regression
output_table = lifetime_analyzer.regression()
output_table.to_csv(output_regression_filename, sep='\t', index=False)
print("Output regression to {0}.".format(output_regression_filename))

if output_graphs:
    output_filename = pathlib.Path(output_folder).joinpath('regression.png')
    print("Output a regression graph to {0}.".format(output_filename))
    figure = pyplot.figure(figsize = (12, 8), dpi = 300)
    axes = figure.add_subplot(111)
    axes.plot(output_table.life_time, output_table.spot_count)
    figure.savefig(output_filename, dpi = 300)

# lifetime
output_table = lifetime_analyzer.lifetime()
output_table.to_csv(output_lifetime_filename, sep='\t', index=False)
print("Output lifetime to {0}.".format(output_lifetime_filename))

if output_graphs:
    output_filename = pathlib.Path(output_folder).joinpath('lifetime.png')
    print("Output a lifetime graph to {0}.".format(output_filename))
    figure = pyplot.figure(figsize = (12, 8), dpi = 300)
    axes = figure.add_subplot(111)
    width = output_table.life_time[0]
    axes.bar(output_table.life_time, output_table.spot_count, width = -width, align = 'edge')
    figure.savefig(output_filename, dpi = 300)

# new binding and lifetime
output_table = lifetime_analyzer.newbinding()
output_table.to_csv(output_newbinding_filename, sep='\t', index=False)
print("Output new bindings to {0}.".format(output_newbinding_filename))

if output_graphs:
    print("Output a new-binding graph to {0}.".format(output_filename))
    output_filename = pathlib.Path(output_folder).joinpath('newbinding.png')
    figure = pyplot.figure(figsize = (12, 8), dpi = 300)
    axes = figure.add_subplot(111)
    axes.scatter(output_table.plane, output_table.life_time)
    figure.savefig(output_filename, dpi = 300)

# cumulative lifetime
output_table = lifetime_analyzer.cumulative()
output_table.to_csv(output_cumulative_filename, sep='\t', index=False)
print("Output cumulative lifetime to {0}.".format(output_cumulative_filename))

if output_graphs:
    print("Output a cumulative graph to {0}.".format(output_filename))
    output_filename = pathlib.Path(output_folder).joinpath('cumulatie.png')
    figure = pyplot.figure(figsize = (12, 8), dpi = 300)
    axes = figure.add_subplot(111)
    axes.scatter(output_table.life_time, output_table.spot_count)
    figure.savefig(output_filename, dpi = 300)
