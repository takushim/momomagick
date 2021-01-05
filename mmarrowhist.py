#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas
from matplotlib import pyplot

# default values
input_filename = None
time_scale = 1.0
output_folder = 'analyze'
suffix_tsv_file = ".txt"
suffix_graph_file = ".png"
stemsuffix_histogram = '_histogram'
xy_resolution = 0.1625

# parse arguments
parser = argparse.ArgumentParser(description='Analyze findarrow output', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-x', '--time-scale', type = float, default=time_scale, \
                    help='interval of time-lapse (in seconds)')

parser.add_argument('input_file', default=input_filename, \
                    help='input TSV file')
args = parser.parse_args()

# set arguments
time_scale = args.time_scale
input_filename = args.input_file
output_filename_stem = pathlib.Path(input_filename).stem
output_filename_stem = re.sub('\.ome$', '', output_filename_stem, flags=re.IGNORECASE)

# prepare a folder
print("Making an output folder {0}.".format(input_filename))
pathlib.Path(output_folder).mkdir(parents = True, exist_ok = True)

# read TSV or TrackJ CSV file
print("Read TSV from {0}.".format(input_filename))
spot_table = pandas.read_csv(input_filename, comment = '#', sep = '\t')

# histogram of binding spots
print("Making a histogram of binding points.")
if "arrow_pos_um" not in spot_table.columns:
    spot_table["arrow_pos_um"] = spot_table.arrow_pos * xy_resolution
    print("Calculating spot position in um assuming {} pixel/um".format(xy_resolution))

bin_range = [-2, 10, 0.5]
given_bins = numpy.arange(*bin_range)
output_hist, output_bins = numpy.histogram(numpy.array(spot_table.arrow_pos_um), bins = given_bins)
output_bins = output_bins[:-1]
print(len(given_bins), len(output_hist), len(output_bins))
print("Histogram: {} spots out of range".format(len(spot_table) - numpy.sum(output_hist)))
columns = ["bin", "count"]
output_table = pandas.DataFrame({columns[0]: output_bins, columns[1]: output_hist}, columns = columns)

output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_histogram + suffix_tsv_file)
print("Output a histogram TSV to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep='\t', index=False)

output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_histogram + suffix_graph_file)
print("Output a histogram graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Distribution of binding", size = 'xx-large')
axes.bar(output_bins, output_hist, width = bin_range[2] * 0.9, align = 'edge', color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            "{} spots out of range".format(len(spot_table) - numpy.sum(output_hist)), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)

print(".")
    
