#!/usr/bin/env python

import sys, pathlib, re, argparse, numpy, pandas
from matplotlib import pyplot
from mmtools import trackj, lifetime

# default values
input_filename = None
time_scale = 1.0
output_folder = 'analyze'
suffix_tsv_file = ".txt"
suffix_graph_file = ".png"
stemsuffix_regression = '_regression'
stemsuffix_lifetime = '_lifetime'
stemsuffix_newbinding = '_newbinding'
stemsuffix_cumulative = '_cumulative'
stemsuffix_histogram = '_histogram'
xy_resolution = 0.1625

# parse arguments
parser = argparse.ArgumentParser(description='Calculate lifetime and regression curves', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-x', '--time-scale', type = float, default=time_scale, \
                    help='interval of time-lapse (in seconds)')

parser.add_argument('input_file', default=input_filename, \
                    help='input TSV file or TrackJ CSV file')
args = parser.parse_args()

# set arguments
time_scale = args.time_scale
input_filename = args.input_file
output_filename_stem = pathlib.Path(input_filename).stem
output_filename_stem = re.sub('\.ome$', '', output_filename_stem, flags=re.IGNORECASE)
output_filename_stem = re.sub('\_speckles$', '', output_filename_stem, flags=re.IGNORECASE)

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
output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_regression + suffix_tsv_file)
output_table = lifetime_analyzer.regression()
curve_func, popt, pcov = lifetime.Lifetime.fit_one_phase_decay(output_table.life_time, output_table.spot_count)
print("Output regression to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep='\t', index=False)
print("One phase decay model: Off-rate = {0:f}, half-life = {1:f}".format(popt[1], numpy.log(2) / popt[1]))

output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_regression + suffix_graph_file)
print("Output a regression graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Regression from t = 0", size = 'xx-large')
axes.plot(output_table.life_time, curve_func(output_table.life_time), color = 'black', linestyle = ':')
axes.scatter(output_table.life_time, output_table.spot_count, color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            "Off-rate = {0:.3f} /sec, Half-life = {1:.3f} sec".format(popt[1], numpy.log(2) / popt[1]), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)

print(".")

# lifetime
output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_lifetime + suffix_tsv_file)
output_table = lifetime_analyzer.lifetime()
curve_func, popt, pcov = lifetime.Lifetime.fit_one_phase_decay(output_table.life_time, output_table.spot_count)

print("Output lifetime to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep='\t', index=False)
print("One phase decay model: Off-rate = {0:f}, half-life = {1:f}".format(popt[1], numpy.log(2) / popt[1]))

output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_lifetime + suffix_graph_file)
print("Output a lifetime graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Lifetime distribution", size = 'xx-large')
width = output_table.life_time[0]
axes.plot(output_table.life_time, curve_func(output_table.life_time), color = 'black', linestyle = ':')
axes.bar(output_table.life_time, output_table.spot_count, width = -width/2, align = 'edge', color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            "Off-rate = {0:.3f} /sec, Half-life = {1:.3f} sec".format(popt[1], numpy.log(2) / popt[1]), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)

print(".")

# cumulative lifetime
output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_cumulative + suffix_tsv_file)
output_table = lifetime_analyzer.cumulative()
curve_func, popt, pcov = lifetime.Lifetime.fit_one_phase_decay(output_table.life_time, output_table.spot_count)

print("Output cumulative lifetime to {0}.".format(output_filename))
output_table.to_csv(output_filename, sep='\t', index=False)
print("One phase decay model: Off-rate = {0:f}, half-life = {1:f}".format(popt[1], numpy.log(2) / popt[1]))

output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_cumulative + suffix_graph_file)
print("Output a cumulative graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Cumulative regression", size = 'xx-large')
axes.plot(output_table.life_time, curve_func(output_table.life_time), color = 'black', linestyle = ':')
axes.scatter(output_table.life_time, output_table.spot_count, color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            "Off-rate = {0:.3f} /sec, Half-life = {1:.3f} sec".format(popt[1], numpy.log(2) / popt[1]), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)

print(".")

# new binding and lifetime
output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_newbinding + suffix_tsv_file)
output_table = lifetime_analyzer.newbinding()
output_table.to_csv(output_filename, sep='\t', index=False)
print("Output new bindings to {0}.".format(output_filename))
mean_lifetime = output_table[output_table.plane > 0].life_time.mean()
print("Mean lifetime = {0} sec (plane > 0).".format(mean_lifetime))

output_filename = pathlib.Path(output_folder).joinpath(output_filename_stem + stemsuffix_newbinding+ suffix_graph_file)
print("Output a new-binding graph to {0}.".format(output_filename))
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Binding plane and lifetime", size = 'xx-large')
axes.axhline(mean_lifetime, color = 'black', linestyle = ':')
axes.scatter(output_table.plane, output_table.life_time, color = 'orange')
axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
            "Mean lifetime = {0:.3f} sec (plane > 0)".format(mean_lifetime), \
            size = 'xx-large', ha = 'right', va = 'top')
figure.savefig(output_filename, dpi = 300)

print(".")
