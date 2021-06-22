#!/usr/bin/env python

import sys, argparse, tifffile
import numpy as np
import pandas as pd
from scipy import ndimage, optimize
from mmtools import mmtiff

# defaults
input_filename = None
output_filename = 'affine.txt'
ref_filename = None
use_channel = 0
output_aligned_image = False
aligned_image_filename = None
aligned_image_suffix = '_affine.tif'
parallel_shift_only = False
gpu_id = None

parser = argparse.ArgumentParser(description='Calculate sample shift using affine matrix and optimization', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--output-file', default = output_filename, \
                    help='output TSV file name ({0} if not specified)'.format(output_filename))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='GPU ID')

parser.add_argument('-r', '--ref-image', default = ref_filename, \
                    help='specify a reference image')

parser.add_argument('-c', '--use-channel', type = int, default = use_channel, \
                    help='specify the channel to process')

parser.add_argument('-p', '--parallel-shift-only', action = 'store_true', \
                    help='Optimize for parallel shift only')

parser.add_argument('-A', '--output-aligned-image', action = 'store_true', \
                    help='output aligned images')

parser.add_argument('-a', '--aligned-image-file', default = aligned_image_filename, \
                    help='filename to output images ([basename]{0} if not specified)'.format(aligned_image_suffix))

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
output_filename = args.output_file
ref_filename = args.ref_image
use_channel = args.use_channel
gpu_id = args.gpu_id
parallel_shift_only = args.parallel_shift_only
output_aligned_image = args.output_aligned_image

if args.aligned_image_file is None:
    aligned_image_filename = mmtiff.with_suffix(input_filename, aligned_image_suffix)
else:
    aligned_image_filename = args.aligned_image_filename

# activate GPU
if gpu_id is not None:
    import cupy as cp
    from cupyx.scipy import ndimage as cpimage
    device = cp.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)

# read input image
input_tiff = mmtiff.MMTiff(input_filename)
if input_tiff.colored:
    raise Exception('Input_image: color image not accepted')

input_images = input_tiff.as_list(channel = use_channel, drop = True)

# read reference image
if ref_filename is None:
    ref_image = input_images[0].copy()
else:
    ref_tiff = mmtiff.MMTiff(ref_filename)
    if ref_tiff.colored:
        raise Exception('Reference image: color reference image not accepted.')
    ref_image = ref_tiff.as_list(channel = use_channel, drop = True)[0]

# hanning window
hanning_z = np.hanning(input_tiff.total_zstack)
hanning_y = np.hanning(input_tiff.height)
hanning_x = np.hanning(input_tiff.width)
mesh_z, mesh_y, mesh_x = np.meshgrid(hanning_z, hanning_y, hanning_x, indexing = 'ij')
hanning_mat = mesh_z * mesh_y * mesh_x

# prepare fft_conj of reference images (center = no drifting)
if gpu_id is None:
    ref_fft_conj = np.conj(np.fft.fftn(ref_image * hanning_mat))
else:
    hanning_mat = cp.array(hanning_mat)
    ref_fft_conj = cp.conj(cp.fft.fftn(cp.array(ref_image) * hanning_mat))

# calculate POCs for pre-registration
poc_image_list = []
shift_list = []
center = np.array(ref_image.shape) // 2
for index in range(len(input_images)):
    if gpu_id is None:
        image = input_images[index]
        image_fft = np.fft.fftn(image * hanning_mat)
        corr_image = ref_fft_conj * image_fft / np.abs(ref_fft_conj * image_fft)
        poc_image = np.fft.fftshift(np.real(np.fft.ifftn(corr_image)))
    else:
        image = cp.array(input_images[index])
        image_fft = cp.fft.fftn(image * hanning_mat)
        corr_image = ref_fft_conj * image_fft / cp.abs(ref_fft_conj * image_fft)
        poc_image = cp.fft.fftshift(cp.real(cp.fft.ifftn(corr_image)))
        poc_image = cp.asnumpy(poc_image)

    max_pos = ndimage.maximum_position(poc_image)
    max_val = poc_image[max_pos]
    image_shift = (max_pos - center).astype(float)
    print("POC performed. Index:", index, "Shift:", image_shift, "Corr:", max_val)
    shift_list.append({'index': index, \
        'init_x': image_shift[2], 'init_y': image_shift[1], 'init_z': image_shift[0], \
        'poc_corr': max_val})

# prepare normalized reference image
def normalize (image):
    clip_max = np.percentile(image, 99)
    clip_min = np.percentile(image, 1)
    return (image.clip(clip_min, clip_max) - clip_min) / (clip_max - clip_min)


if gpu_id is None:
    ref_float = normalize(ref_image)
else:
    ref_float = cp.array(normalize(ref_image))

matrix_list = []
output_image_list = []
# optimization using an affine matrix
# note: input = matrix * output + offset
for index in range(len(input_images)):
    print("Starting optimization:", index)
    init_x = shift_list[index]['init_x']
    init_y = shift_list[index]['init_y']
    init_z = shift_list[index]['init_z']
    if parallel_shift_only:
        print("Parallel shift only. Freedom = 3.")
        if input_tiff.total_zstack == 1:
            input_image = input_images[index][0]
            init_params = np.array([init_y, init_x])
            params_to_matrix = lambda params: np.array([[1.0, 0.0, params[0]], [0.0, 1.0, params[1]], [0.0, 0.0, 1.0]])
        else:
            input_image = input_images[index]
            init_params = np.array([init_z, init_y, init_x])
            params_to_matrix = lambda params: np.array([[1.0, 0.0, 0.0, params[0]], [0.0, 1.0, 0.0, params[1]], \
                                                        [0.0, 0.0, 1.0, params[2]], [0.0, 0.0, 0.0, 1.0]])
    else:
        print("Full affine transformation. Freedom = 12.")
        if input_tiff.total_zstack == 1:
            input_image = input_images[index][0]
            init_params = np.array([1.0, 0.0, init_y, 0.0, 1.0, init_x])
            params_to_matrix = lambda params: np.array([params[0:3], params[3:6], [0.0, 0.0, 1.0]])
        else:
            input_image = input_images[index]
            init_params = np.array([1.0, 0.0, 0.0, init_z, 0.0, 1.0, 0.0, init_y, 0.0, 0.0, 1.0, init_x])
            params_to_matrix = lambda params: np.array([params[0:4], params[4:8], params[8:12], [0.0, 0.0, 0.0, 1.0]])

    if gpu_id is None:
        image_float = normalize(input_image)
        def error_func (params):
            matrix = params_to_matrix(params)
            trans_float = ndimage.affine_transform(image_float, matrix)
            error = np.sum((ref_float - trans_float) * (ref_float - trans_float) * hanning_mat)
            return error
    else:
        image_float = cp.array(normalize(input_image))
        def error_func (params):
            matrix = cp.array(params_to_matrix(params))
            trans_float = cpimage.affine_transform(image_float, matrix)
            error = cp.asnumpy(cp.sum((ref_float - trans_float) * (ref_float - trans_float) * hanning_mat))
            return error

    results = optimize.minimize(error_func, init_params, method = "Powell")
    final_matrix = params_to_matrix(results.x)

    if output_aligned_image:
        if input_tiff.total_zstack == 1:
            input_image = input_images[index][0]
        else:
            input_image = input_images[index]

        if gpu_id is None:
            aligned_image = ndimage.affine_transform(input_image, final_matrix)
        else:
            output_image = cpimage.affine_transform(cp.array(input_image), cp.array(final_matrix))
            output_image = cp.asnumpy(output_image)

        if input_tiff.total_zstack == 1:
            output_image = output_image[np.newaxis, np.newaxis]
        else:
            output_image = output_image[:, np.newaxis]

        output_image_list.append(output_image)

    shift_list[index]['matrix'] = ','.join([str(x) for x in results.x])
    print(results.message, "Matrix:")
    print(final_matrix)
    print(".")

# open tsv file and write header
print("Output TSV:", output_filename)
shift_table = pd.DataFrame(shift_list)
shift_table.to_csv(output_filename, sep = '\t', index = False)

# output images
if output_aligned_image:
    print("Output image:", aligned_image_filename)
    input_tiff.save_image_ome(aligned_image_filename, np.array(output_image_list))
