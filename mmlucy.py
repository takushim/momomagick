#!/usr/bin/env python

import sys, argparse, tifffile, time
import numpy as np
from pathlib import Path
from mmtools import mmtiff, register, lucy, gpuimage

# defaults
psf_folder = Path(__file__).parent.joinpath('psf')
input_filename = None
output_filename = None
output_suffix = '_dec.tif'
psf_filename = 'diSPIM.tif'
iterations = 10
gpu_id = None

parser = argparse.ArgumentParser(description = 'Deconvolve images using the Richardson-Lucy algorhythm', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} if not specified)'.format(output_suffix))

parser.add_argument('-p', '--psf-image', default = psf_filename, \
                    help='filename of psf image, searched in current folder -> program folder')

parser.add_argument('-i', '--iterations', default = iterations, \
                    help='number of iterations')

parser.add_argument('-s', '--z-scale-psf', action = 'store_true', \
                    help='Scale the z axix of psf (not image)')

parser.add_argument('-r', '--restore-z-scale', action = 'store_true', \
                    help='Restore z scaling of images after deconvolution')

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='Turn on GPU use with the specified ID')

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to deconvolve')

args = parser.parse_args()

# defaults
iterations = args.iterations
psf_filename = args.psf_image
restore_z_scale = args.restore_z_scale
z_scale_psf = args.z_scale_psf
gpu_id = args.gpu_id
input_filename = args.input_file
if args.output_file is None:
    output_filename = mmtiff.with_suffix(input_filename, output_suffix)
else:
    output_filename = args.output_file

# turn on GPU device
lucy.turn_on_gpu(gpu_id)

# load psf image
if Path(psf_filename).exists():
    print("Read PSF image in the current folder:", psf_filename)
    psf_image = tifffile.imread(psf_filename)
else:
    psf_path = Path(psf_folder).joinpath(psf_filename)
    if psf_path.exists():
        print("Read PSF image in the system folder:", str(psf_path))
        psf_image = tifffile.imread(str(psf_path))
    else:
        raise Exception('PSF file {0} not found'.format(psf_filename))

# load input image
input_tiff = mmtiff.MMTiff(input_filename)
input_list = input_tiff.as_list(list_channel = True)

# setting image scale
z_scale_ratio = 1
if input_tiff.total_zstack > 1:
    ratio = input_tiff.z_step_um / input_tiff.pixelsize_um
    if np.isclose(ratio, 1.0) is False:
        print("Setting z scaling of images:", ratio)
        z_scale_ratio = ratio
else:
    z_scale_ratio = 1

# z-zoom psf
if z_scale_psf and input_tiff.total_zstack > 1:
    ratio = input_tiff.pixelsize_um / input_tiff.z_step_um
    if np.isclose(ratio, 1.0) is False:
        psf_image = gpuimage.z_zoom(psf_image, ratio, gpu_id = gpu_id)
        print("Scaling psf image into:", ratio)

# deconvolve
deconvolver = lucy.Lucy(psf_image, gpu_id)

# save results in the CTZYX order
output_image_list = []
print("Start deconvolution:", time.ctime())
print("Frames:", end = ' ')
for index in range(input_tiff.total_time):
    image_list = []
    for channel in range(input_tiff.total_channel):
        image = input_list[index][channel]
        if z_scale_ratio != 1:
            image = gpuimage.z_zoom(image, z_scale_ratio, gpu_id = gpu_id)
        image = deconvolver.deconvolve(image, iterations)
        if restore_z_scale:
            image = gpuimage.z_zoom(image, 1 / z_scale_ratio, gpu_id = gpu_id)
        image_list.append(image)
    output_image_list.append(image_list)
    print(index, end = ' ', flush = True)
print(".")
print("End deconvolution:", time.ctime())

# shape output into the TZCYX order
output_image = np.array(output_image_list).swapaxes(1, 2)
if (input_tiff.dtype.kind == 'i' or input_tiff.dtype.kind == 'u') and \
        np.max(output_image) <= np.iinfo(input_tiff.dtype).max:
    output_image = mmtiff.float_to_int(output_image, input_tiff.dtype)
else:
    output_image = output_image.astype(np.float32)

# output in the ImageJ format, dimensions should be in TZCYX order
input_tiff.save_image(output_filename, output_image)
