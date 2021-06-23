#!/usr/bin/env python

import sys, argparse
import numpy as np
import pandas as pd
from mmtools import mmtiff, regist

# defaults
input_filename = None
tsv_filename = None
tsv_suffix = '_affine.txt'
output_filename = None
output_suffix = '_reg.tif'
gpu_id = None

parser = argparse.ArgumentParser(description='Regist time-lapse images using affine matrices', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} if not specified)'.format(output_suffix))

parser.add_argument('-f', '--tsv-file', default = tsv_filename, \
                    help='filename to output images ([basename]{0} if not specified)'.format(tsv_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
gpu_id = args.gpu_id
if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

if args.tsv_file is None:
    tsv_filename = mmtiff.with_suffix(input_filename, tsv_suffix)
else:
    tsv_filename = args.tsv_file

# activate GPU
if gpu_id is not None:
    regist.turn_on_gpu(gpu_id)

# read input image
input_tiff = mmtiff.MMTiff(input_filename)
if input_tiff.colored:
    raise Exception('Input_image: color image not accepted')
input_images = input_tiff.as_list()

# read TSV matrices and convert them to a list of dicts
summary_list = pd.read_csv(tsv_filename, comment = '#', sep = '\t').to_dict('records')

# registration using affine matrices
# note: input = matrix * output + offset
output_image_list = []
print("Transforming image:", end = ' ')
for summary in summary_list:
    index = summary['index']
    print(index, end = ' ', flush = True)
    input_image = input_images[index]

    matrix = regist.csv_to_matrix(summary['aff_mat'])

    channel_image_list = []
    for channel in range(input_tiff.total_channel):
        if input_tiff.total_zstack == 1:
            input_image = input_images[index][0, channel]
            output_image = regist.affine_transform(input_image, matrix, gpu_id)
            output_image = output_image[np.newaxis]
        else:
            input_image = input_images[index][:, channel]
            output_image = regist.affine_transform(input_image, matrix, gpu_id)
        channel_image_list.append(output_image)
    output_image_list.append(channel_image_list)
print(".")

# Swap channel and z axis
output_images = np.array(output_image_list)
output_images = np.swapaxes(output_images, 2, 1)

# output images
if len(output_image_list) > 0:
    print("Output image:", output_filename)
    input_tiff.save_image_ome(output_filename, output_images)
else:
    print("No output image")