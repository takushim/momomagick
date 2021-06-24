#!/usr/bin/env python

import sys, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mmtools import mmtiff, regist

# default values
input_filename = None
output_txt_filename = None
output_txt_suffix = '_analyze.txt'
output_image = False
output_image_filename = None
output_image_suffix = '_analyze.png'

# parse arguments
parser = argparse.ArgumentParser(description='Decompose the registration affine matrices ', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-txt-file', default = output_txt_filename, \
                    help='output TSV text file name ([basename]{0} if not specified)'.format(output_txt_suffix))

parser.add_argument('-I', '--output-image', action = 'store_true', \
                    help='output graph images of decomposed affine matrices')

parser.add_argument('-i', '--output-image-file', default = output_image_filename, \
                    help='filename to output graph images ([basename]{0} if not specified)'.format(output_image_suffix))

parser.add_argument('input_file', default=input_filename, \
                    help='input JSON file recording affine matrices')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
output_image = args.output_image
if args.output_txt_file is None:
    output_txt_filename = mmtiff.with_suffix(input_filename, output_txt_suffix)
else:
    output_txt_filename = args.output_txt_file

if args.output_image_file is None:
    output_image_filename = mmtiff.with_suffix(input_filename, output_image_suffix)
else:
    output_image_filename = args.output_image_file

# read the JSON file
with open(input_filename, 'r') as f:
    print("Reading JSON file:", input_filename)
    json_data = json.load(f)
    parameters = json_data['parameters']
    for key, value in parameters.items():
        print(key, value)
    summary_list = json_data['summary_list']

# prepare a pandas table
record_list = []
output_affine_params = ['transport', 'rotation_angles', 'zoom', 'shear']
short_affine_params = ['trans', 'rot', 'zoom', 'shear']
for summary in summary_list:
    if 'decomposed' not in summary['affine']:
        summary['affine']['decomposed'] = regist.decompose_matrix(summary['affine']['matrix'])
    record = {}
    record['index'] = summary['index']
    record['poc'] = summary['poc']['shift']
    record['init'] = summary['affine']['init']
    for output_param, short_param in zip(output_affine_params, short_affine_params):
        record[short_param] = summary['affine']['decomposed'][output_param]
    record_list.append(record)

table_record_list = []
for record in record_list:
    table_record = {}
    table_record['index'] = record['index']
    for key in [x for x in record.keys() if x != 'index']:
        for pos, value in zip(['x', 'y', 'z'], record[key][::-1]):
            table_record["{0}_{1}".format(key, pos)] = value
    table_record_list.append(table_record)

# output the pandas table
summary_table = pd.DataFrame(table_record_list).sort_values(by = 'index')
print("Output TSV file:", output_txt_filename)
summary_table.to_csv(output_txt_filename, sep = '\t', index = False)

# output graphs
if output_image:
    figure = plt.figure(figsize = (12, 8), dpi = 300)
    x_values = np.array([x['index'] for x in record_list])
    plot_list = ['poc', 'init'] + short_affine_params
    title_list = ['poc shift', 'initial shift'] + output_affine_params

    for index in range(len(plot_list)):
        axes = figure.add_subplot(2, 3, index + 1, title = title_list[index])
        y_values = np.array([x[plot_list[index]][::-1] for x in record_list])
        y_labels = ['x', 'y', 'z'][:len(y_values[1])]
        axes.plot(x_values, y_values, label = y_labels)
        handles, labels = axes.get_legend_handles_labels()

    figure.legend(handles, labels, loc = 'center right')

    print("Output graph image:", output_image_filename)
    figure.savefig(output_image_filename)