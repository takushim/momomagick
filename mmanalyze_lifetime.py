#!/usr/bin/env python

import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from mmtools import stack, trackj, lifetime, particles, log

# default values
input_filenames = None
time_scale = None
fitting_start = 0
fitting_end = 0
output_filename = None
output_suffix = "_{0}.txt"
start_plane = 0
graph_filename = None
graph_suffix = "_{0}.png"
analysis_list = ['lifetime', 'regression', 'cumulative', 'scatter']
analysis = analysis_list[0]
time_scale = 1.0
opt_method = lifetime.default_method
opt_method_list = lifetime.optimizing_methods
bleach_frame = 11.95
mean_width = None

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

parser.add_argument('-l', '--start-plane', type = int, default = start_plane, \
                    help='starting plane')

parser.add_argument('-s', '--fitting-start', type = int, default = fitting_start, \
                    help='starting point of fitting')

parser.add_argument('-e', '--fitting-end', type = int, default = fitting_end, \
                    help='end point of fitting. Zero = +Inf.')

parser.add_argument('-m', '--mean-width', type = int, default = mean_width, \
                    help='the width of averaging in the scatter graph (None for auto)')

parser.add_argument('-b', '--bleach-frame', type = float, default = bleach_frame, \
                    help='bleaching rate of the dye (specify by a frame count)')

parser.add_argument('-t', '--opt-method', type = str, default = opt_method, choices = opt_method_list, \
                    help='Method to optimize the one-phase-decay model')

log.add_argument(parser)

parser.add_argument('input_files', nargs = '+', default = input_filenames, \
                    help='input JSON file of tracking data. Results from multiple files are merged.')
args = parser.parse_args()

# logging
logger = log.get_logger(__file__, level = args.log_level)

# set arguments
time_scale = args.time_scale
input_filenames = args.input_files
fitting_start = args.fitting_start
fitting_end = args.fitting_end
opt_method = args.opt_method
analysis = args.analysis
bleach_frame = args.bleach_frame
start_plane = args.start_plane
mean_width = args.mean_width

output_suffix = output_suffix.format(analysis)
graph_suffix = graph_suffix.format(analysis)

output_filename = args.output_file
if output_filename is None:
    output_filename = stack.with_suffix(input_filenames[0], output_suffix)

graph_filename = args.graph_file
if graph_filename is None:
    graph_filename = stack.with_suffix(input_filenames[0], graph_suffix)

# read JSON or TSV or TrackJ CSV file
spot_tables = []
plane_counts = []
for input_filename in input_filenames:
    suffix = Path(input_filename).suffix.lower()
    if suffix == '.json':
        with open(input_filename, 'r') as f:
            json_data = json.load(f)
            plane_count = json_data['image_properties']['t_count']
            spot_table = particles.list_to_table(json_data.get('spot_list', []))
            spot_table['plane'] = spot_table['time']
            spot_table['total_index'] = spot_table['track']
    elif suffix == ".txt":
        spot_table = pd.read_csv(input_filename, comment = '#', sep = '\t')
        plane_count = spot_table['plane'].max()
    elif suffix == ".csv":
        spot_table = trackj.read_spots(input_filename)
        plane_count = spot_table['plane'].max()
    else:
        raise Exception("Unknown file format: {0}".format(input_filename))
    
    total_records = len(spot_table.total_index)
    total_tracks = len(spot_table.total_index.unique())
    logger.info("{0}: {1} records and {2} tracks in {3} frames.".format(input_filename, total_records, total_tracks, plane_count))
    spot_tables.append(spot_table)
    plane_counts.append(plane_count)

# analysis
if analysis == 'regression':
    results = [lifetime.regression(table, time_scale = time_scale) for table in spot_tables]
elif analysis == 'lifetime':
    results = [lifetime.lifetime(table, time_scale = time_scale, start_plane = start_plane) for table in spot_tables]
elif analysis == 'cumulative':
    results = [lifetime.cumulative(table, time_scale = time_scale, start_plane = start_plane) for table in spot_tables]
elif analysis == 'scatter':
    results = [lifetime.new_bindings(table, time_scale = time_scale) for table in spot_tables]
else:
    raise Exception('Unknown analysis method: {0}'.format(analysis))

if analysis in ['regression', 'lifetime', 'cumulative']:
    max_index = np.argmax([len(result.frame) for result in results])
    frames = results[max_index].frame.to_list()
    times = results[max_index].time.to_list()
    times_lower = [index * time_scale for index in range(len(frames))]

    result_dict = {'frame': frames, 'time': times, 'time_lower': times_lower}
    counts_sum = [0] * len(frames)
    for index in range(len(results)):
        counts = [0] * len(frames)
        max_len = len(results[index].spotcount)
        counts[0:max_len] = results[index].spotcount.to_list()
        result_dict['Result_{0}'.format(index)] = counts
        counts_sum = [count + sum for count, sum in zip(counts, counts_sum)]

    result_table = pd.DataFrame(result_dict)
    result_table['sum'] = counts_sum

    if analysis == 'lifetime':
        result_table['sum_ratio'] = [count_sum / sum(counts_sum) for count_sum in counts_sum]
    else:
        result_table['sum_ratio'] = [count_sum / counts_sum[0] for count_sum in counts_sum]

    # fitting
    fitting = lifetime.fit_one_phase_decay(times, counts_sum, start = fitting_start, \
                                        end = fitting_end, method = opt_method)
    fitting_func = fitting['func']
    logger.info("Fitting: {0}".format(fitting['message']))

    # tsv
    logger.info("Output {0} table to {1}.".format(analysis, output_filename))
    result_table.to_csv(output_filename, sep = '\t', index = False)

    # graph
    logger.info("Output {0} graph to {1}.".format(analysis, graph_filename))

    if analysis == 'lifetime':
        graph_title = "Lifetime distribution (total {0} spots)".format(sum(counts_sum))
    elif analysis == 'cumulative':
        graph_title = "Cumulative lifetime (total {0} spots)".format(counts_sum[0])
    else:
        graph_title = "Regression from t = 0 (total {0} spots)".format(counts_sum[0])


    figure = pyplot.figure(figsize = (12, 8), dpi = 300)
    axes = figure.add_subplot(111)
    axes.set_title(graph_title, size = 'xx-large')

    offset = np.zeros_like(times, dtype = float)
    for index in range(len(results)):
        bar_w = times[0] * 0.8
        counts = np.array(result_dict['Result_{0}'.format(index)])
        axes.bar(times, counts, bottom = offset, width = bar_w, label = Path(input_filenames[index]).name)
        offset += np.array(counts)

    curve_x = np.arange(times[0], np.max(times), times[0] / 10)
    curve_y = fitting_func(curve_x)
    axes.plot(curve_x, curve_y, color = 'black', linestyle = ':')
    axes.set_ylim(bottom = 0)

    bleach_rate = np.log(2) / (bleach_frame * time_scale)
    bleach_y = fitting_func(curve_x)[0] * np.exp(-(curve_x - curve_x[0]) * bleach_rate)
    axes.plot(curve_x, bleach_y, color = 'black', linestyle = '-')
    bleach_text = "Bleach = {0:.3e} /sec, Half-life = {1:.3f} sec ({2:.3f} frames)".\
                  format(bleach_rate, bleach_frame * time_scale, bleach_frame)
    logger.info(bleach_text)

    koff = fitting['koff']
    halflife = fitting['halflife']
    fitting_text = "Off-rate = {0:.3e} /sec, Half-life = {1:.3f} sec ({2:.3f} frames)".\
                   format(koff, halflife, halflife / time_scale)
    condition_text = "Fit using {0} <= t <= {1}. Using Plane >= {2}".format(fitting['start'], fitting['end'], start_plane)
    logger.info(fitting_text)
    logger.info(condition_text)

    axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.55, \
              fitting_text, size = 'large', ha = 'right', va = 'top')
    axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.5, \
              bleach_text, size = 'large', ha = 'right', va = 'top')
    axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.45, \
              condition_text, size = 'large', ha = 'right', va = 'top')

    axes.legend()

    figure.savefig(graph_filename, dpi = 300)

elif analysis == 'scatter':
    mean_lifetime = np.mean([result.lifetime.mean() for result in results])
    mean_text = "Mean lifetime = {0:.3f} sec ({1:.3f} frames, plane > 0)".\
                format(mean_lifetime, mean_lifetime / time_scale)
    logger.info(mean_text)

    # add offset and concat
    for index in range(len(results)):
        offset = sum(plane_counts[0:index])
        results[index]['plane'] = results[index]['plane'] + offset
    output_table = pd.concat(results)

    # tsv
    logger.info("Output {0} table to {1}.".format(analysis, output_filename))
    output_table.to_csv(output_filename, sep = '\t', index = False)

    # graph
    logger.info("Output {0} graph to {1}.".format(analysis, graph_filename))
    graph_title = "Relationship between binding and lifetime (total {0} spots)".format(len(output_table))

    figure = pyplot.figure(figsize = (12, 8), dpi = 300)
    axes = figure.add_subplot(111)
    axes.set_title(graph_title, size = 'xx-large')
    axes.axhline(mean_lifetime, color = 'black', linestyle = ':')

    max_plane = output_table['plane'].max()
    max_frame = output_table['lifeframe'].max()

    # plot means
    if mean_width is None:
        mean_width = plane_counts[0] // 10
        logger.info("Automatically setting the averaging width: {0}".format(mean_width))
    else:
        logger.info("Setting the averaging width: {0}".format(mean_width))

    for index in range(max_plane // mean_width + 1):
        plane_min = mean_width * index
        plane_max = min(mean_width * (index + 1) - 1, max_plane)

        plot_table = output_table[(plane_min <= output_table.plane) & (output_table.plane <= plane_max)].reset_index(drop = True)
        if len(plot_table) == 0:
            continue

        plot_mean = plot_table['lifetime'].mean()
        axes.hlines(plot_mean, plane_min, plane_max, color = 'black', linestyle = '-')

    # draw boundaries between series
    max_life = output_table['lifetime'].max()
    for index in range(1, len(results)):
        bound_x = sum(plane_counts[0:index]) + 0.5
        axes.vlines(bound_x, 0, max_life, color = 'gray', linestyle = ':')

    # plot each lifetime
    delta = 0.2
    cmap = pyplot.get_cmap("tab10")
    plane_list = output_table['plane'].to_list()
    plane_zigzag = [plane_list[i] + delta * ((i + 1) // 2) * (-1)**i for i in range(len(plane_list))]
    output_table['plane_zigzag'] = plane_zigzag
    output_table['plot_color'] = [cmap(plane % cmap.N) for plane in plane_list]
    axes.scatter(output_table.plane_zigzag, output_table.lifetime, color = output_table.plot_color, marker = '.')

    axes.text(axes.get_xlim()[1] * 0.95, axes.get_ylim()[1] * 0.95, \
              mean_text, size = 'xx-large', ha = 'right', va = 'top')

    axes.set_xlim(0, sum(plane_counts) - 1)
    figure.savefig(graph_filename, dpi = 300)

else:
    raise Exception('Unknown analysis: {0}'.format(analysis))
