#!/usr/bin/env python

import sys, pathlib, argparse, numpy, pandas
from matplotlib import pyplot
from mmtools import trackj, spotanalyzer

# algorhithms
trackj_handler = trackj.TrackJ()
analyzer = spotanalyzer.SpotAnalyzer()

# default values
input_filename = None
time_scale = 1.0
output_folder = 'spot_graphs'

# parse arguments
parser = argparse.ArgumentParser(description='Make graphs using a TrackerJ CSV file', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-folder', type = str, default = output_folder, \
                    help='a folder to output graphs')
parser.add_argument('-x', '--time-scale', type = float, default = time_scale, \
                    help='interval of time-lapse (in seconds)')
parser.add_argument('input_file', default = input_filename, \
                    help='an input TrackJ CSV file')
args = parser.parse_args()

# set arguments
output_folder = args.output_folder
time_scale = args.time_scale
input_filename = args.input_file

# make an output folder
print("Making an output folder {0}.".format(input_filename))
pathlib.Path(output_folder).mkdir(parents = True, exist_ok = True)

# read TrackJ CSV file
print("Read TrackJ CSV from {0}.".format(input_filename))
spot_table = trackj_handler.read_spots(input_filename)

# regression graph
output_filename = pathlib.Path(output_folder).joinpath('regression.png')
output_table = analyzer.regression(spot_table, time_scale)

print("Output a regression graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.plot(output_table.life_time, output_table.spot_count)
figure.savefig(output_filename, dpi = 300)

# lifetime
output_filename = pathlib.Path(output_folder).joinpath('lifetime.png')
output_table = analyzer.lifetime(spot_table, time_scale)

print("Output a lifetime graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.bar(output_table.life_time, output_table.spot_count)
figure.savefig(output_filename, dpi = 300)

# new binding and lifetime
output_filename = pathlib.Path(output_folder).joinpath('newbinding.png')
output_table = analyzer.new_binding(spot_table, time_scale)

print("Output a new-binding graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.scatter(output_table.plane, output_table.life_time)
figure.savefig(output_filename, dpi = 300)
output_table.to_csv("debug.csv")

