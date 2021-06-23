#!/usr/bin/env python

import sys, argparse
import numpy as np
import pandas as pd
from scipy import ndimage
from mmtools import mmtiff
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
except ImportError:
    pass

# defaults
input_filename = None
tsv_filename = None
tsv_suffix = '_affine.txt'
output_filename = None
output_suffix = '_reg.tif'
gpu_id = None

parser = argparse.ArgumentParser(description='Regist images using calculated affine matrices', \
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
    device = cp.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)

# read input image
input_tiff = mmtiff.MMTiff(input_filename)
if input_tiff.colored:
    raise Exception('Input_image: color image not accepted')
input_images = input_tiff.as_list()

# read TSV matrices and convert them to a list of dicts
affine_list = pd.read_csv(tsv_filename, comment = '#', sep = '\t').to_dict('records')

# registration using affine matrices
# note: input = matrix * output + offset
output_image_list = []
for record in affine_list:
    index = record['index']
    print("Transforming image:", index)
    input_image = input_images[index]

    matrix_list = [float(x) for x in record['matrix'].split(',')]
    if input_tiff.total_zstack == 1:
        if len(matrix_list) == 6:
            matrix = np.array(matrix_list + [0.0, 0.0, 1.0]).reshape(3, 3)
        elif len(matrix_list) == 9:
            matrix = np.array(matrix_list).reshape(3, 3)
        else:
            print("Cannot transform 3D images using a 3x3 matrix. Skipping.")
            continue
    else:
        if len(matrix_list) == 12:
            matrix = np.array(matrix_list + [0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
        elif len(matrix_list) == 16:
            matrix = np.array(matrix_list).reshape(4, 4)
        else:
            print("Cannot transform 2D images using a 4x4 matrix. Skipping.")
            continue

    if gpu_id is None:
        input_image = input_images[index]
        if input_tiff.total_zstack == 1:
            output_image = np.zeros_like(input_image[0])
            for channel in range(input_tiff.total_channel):
                output_image[channel] = ndimage.affine_transform(input_image[channel], matrix)
            output_image = output_image[np.newaxis]
        else:
            output_image = np.zeros_like(input_image)
            for channel in range(input_tiff.total_channel):
                output_image[:, channel] = ndimage.affine_transform(input_image[:, channel], matrix)
    else:
        input_image = cp.array(input_images[index])
        if input_tiff.total_zstack == 1:
            output_image = cp.zeros_like(input_image[0])
            for channel in range(input_tiff.total_channel):
                output_image[channel] = cpimage.affine_transform(input_image[channel], cp.array(matrix))
        else:
            output_image = cp.zeros_like(input_image)
            for channel in range(input_tiff.total_channel):
                output_image[:, channel] = cpimage.affine_transform(input_image[:, channel], cp.array(matrix))
        output_image = cp.asnumpy(output_image)
        output_image = output_image[np.newaxis]

    output_image_list.append(output_image)

# output images
if len(output_image_list) > 0:
    print("Output image:", output_filename)
    input_tiff.save_image_ome(output_filename, np.array(output_image_list))
else:
    print("No output image")