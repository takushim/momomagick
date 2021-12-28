#!/usr/bin/env python

import sys, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from mmtools import mmtiff, trackj, lifetime, particles

# default values
input_filenames = None
time_scale = None
fitting_start = 0
output_filename = None
output_suffix = "_{0}.txt"
graph_filename = None
graph_suffix = "_{0}.png"
analysis_list = ['lifetime', 'regression', 'cumulative', 'counting']
analysis = analysis_list[0]
time_scale = 1.0
opt_method = lifetime.default_method
opt_method_list = lifetime.optimizing_methods

# parse arguments
parser = argparse.ArgumentParser(description='Calculate lifetime and regression curves', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output filenames ([basename]_{0} if not specified)'.format(output_suffix.format('[analysis]')))

parser.add_argument('-g', '--graph-file', default = graph_filename, \
                    help='graph filenames ([basename]_{0} if not specified)'.format(graph_suffix.format('[analysis]')))

parser.add_argument('-a', '--analysis', type = str, default = analysis, choices = analysis_list, \
                    help='Method used to analyze the tracking data')

parser.add_argument('-x', '--time-scale', type = float, default = time_scale, \
                    help='interval of time-lapse (in seconds)')

parser.add_argument('-s', '--fitting-start', type = int, default = fitting_start, \
                    help='starting point of fitting')

parser.add_argument('-t', '--opt-method', type = str, default = opt_method, choices = opt_method_list, \
                    help='Method to optimize the one-phase-decay model')

parser.add_argument('input_files', nargs = '+', default = input_filenames, \
                    help='input JSON file of tracking data. Results from multiple files are merged.')
args = parser.parse_args()

# set arguments
time_scale = args.time_scale
input_filenames = args.input_files
fitting_start = args.fitting_start
opt_method = args.opt_method
analysis = args.analysis

output_suffix = output_suffix.format(analysis)
graph_suffix = graph_suffix.format(analysis)

output_filename = args.output_file
if output_filename is None:
    output_filename = mmtiff.with_suffix(input_filenames[0], output_suffix)

graph_filename = args.graph_file
if graph_filename is None:
    graph_filename = mmtiff.with_suffix(input_filenames[0], graph_suffix)

# read JSON or TSV or TrackJ CSV file
spot_tables = []
for input_filename in input_filenames:
    suffix = Path(input_filename).suffix.lower()
    if suffix == '.json':
        with open(input_filename, 'r') as f:
            spot_table = pd.DataFrame(particles.parse_tree(json.load(f)['spot_list']))
            spot_table['plane'] = spot_table['time']
            spot_table['total_index'] = spot_table['track']
    elif suffix == ".txt":
        spot_table = pd.read_csv(input_filename, comment = '#', sep = '\t')
    elif suffix == ".csv":
        spot_table = trackj.read_spots(input_filename)
    else:
        raise Exception("Unknown file format.")
    
    total_records = len(spot_table.total_index)
    total_tracks = len(spot_table.total_index.unique())
    print("{0}: {1} records and {2} tracks.".format(input_filename, total_records, total_records))
    spot_tables.append(spot_table)

# analysis
if analysis == 'regression':
    results = [lifetime.regression(table, time_scale = time_scale) for table in spot_tables]
elif analysis == 'lifetime':
    results = [lifetime.lifetime(table, time_scale = time_scale) for table in spot_tables]
elif analysis == 'cumulative':
    results = [lifetime.cumulative(table, time_scale = time_scale) for table in spot_tables]
elif analysis == 'counting':
    results = [lifetime.new_bindings(table, time_scale = time_scale) for table in spot_tables]
else:
    raise Exception('Unknown analysis method: {0}'.format(analysis))

if analysis != 'counting':
    max_index = np.argmax([len(result.frame) for result in results])
    frames = results[max_index].frame.to_list()
    times = results[max_index].time.to_list()

    result_dict = {'frame': frames, 'time': times}
    counts_sum = [0] * len(frames)
    for index in range(len(results)):
        counts = [0] * len(frames)
        max_len = len(results[index].spotcount)
        counts[0:max_len] = results[index].spotcount.to_list()
        result_dict['Result_{0}'.format(index)] = counts
        counts_sum = [count + sum for count, sum in zip(counts, counts_sum)]

    result_table = pd.DataFrame(result_dict)
    result_table['sum'] = counts_sum

    # fitting
    fitting = lifetime.fit_one_phase_decay(times, counts_sum, start = fitting_start, method = opt_method)
    fitting_func = fitting['func']
    print("Params:", fitting['params'])
    print("Status:", fitting['message'])

    # tsv
    print("Output {0} table to {1}.".format(analysis, output_filename))
    result_table.to_csv(output_filename, sep = '\t', index = False)

    # graph
    print("Output {0} graph to {1}.".format(analysis, graph_filename))

    if analysis == 'regression':
        graph_title = 'Regression from t = 0'
    else:
        graph_title = '{0} ({1} spots)'.format(analysis, sum(counts_sum))

    figure = pyplot.figure(figsize = (12, 8), dpi = 300)
    axes = figure.add_subplot(111)
    axes.set_title(graph_title, size = 'xx-large')
    curve_x = np.arange(times[0], np.max(times), times[0] / 10)
    axes.plot(curve_x, fitting_func(curve_x), color = 'black', linestyle = ':')

    offset = np.zeros_like(times, dtype = float)
    for index in range(len(results)):
        counts = np.array(result_dict['Result_{0}'.format(index)])
        axes.bar(times, counts, bottom = offset, width = times[0] / 2, label = input_filenames[index])
        offset += np.array(counts)

    koff = fitting['koff']
    halflife = fitting['halflife']
    start = fitting['start']
    fitting_text = "Off-rate = {0:.3f} /sec, Half-life = {1:.3f} sec (t >= {2})".format(koff, halflife, start)

    axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.5, \
              fitting_text, size = 'large', ha = 'right', va = 'top')
    axes.legend()

    figure.savefig(graph_filename, dpi = 300)

else:
    output_table = pd.concat(results)
    mean_lifetime = output_table[output_table.plane > 0].lifetime.mean()

    # tsv
    print("Output {0} table to {1}.".format(analysis, output_filename))
    output_table.to_csv(output_filename, sep = '\t', index = False)

    # graph
    print("Output {0} graph to {1}.".format(analysis, graph_filename))
    figure = pyplot.figure(figsize = (12, 8), dpi = 300)
    axes = figure.add_subplot(111)
    axes.set_title("Binding plane and lifetime", size = 'xx-large')
    axes.axhline(mean_lifetime, color = 'black', linestyle = ':')
    axes.scatter(output_table.plane, output_table.lifetime, color = 'orange')
    axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
                "Mean lifetime = {0:.3f} sec (plane > 0)".format(mean_lifetime), \
                size = 'xx-large', ha = 'right', va = 'top')
    figure.savefig(graph_filename, dpi = 300)
