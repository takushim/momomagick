#!/usr/bin/env python

import sys, argparse, pathlib, numpy, tifffile, time
from mmtools import mmtiff, lucy

# defaults
psf_folder = pathlib.Path(__file__).parent.joinpath('psf')
input_filename = None
output_filename = None
output_suffix = '_dec.tif'
time_range = [0, 0]
psf_filename = 'diSPIM.tif'
iterations = 10
z_zoom_image = False
z_shrink_image = False
gpu_id = None
use_fft = False
save_memory = False

parser = argparse.ArgumentParser(description='Deconvolve images using the Richardson-Lucy algorhythm', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} if not specified)'.format(output_suffix))

parser.add_argument('-p', '--psf-image', default = psf_filename, \
                    help='filename of psf image, searched in current folder -> program folder')

parser.add_argument('-n', '--number-of-iterations', default = iterations, \
                    help='number of iterations')

parser.add_argument('-z', '--z-zoom-image', action = 'store_true', default = z_zoom_image, \
                    help='Zoom z dimension to achieve xy_scale = z_scale')

parser.add_argument('-s', '--z-shrink-image', action = 'store_true', default = z_shrink_image, \
                    help='Shrink z dimension after deconvolution')

parser.add_argument('-t', '--time-range', nargs = 2, type = int, default = time_range, \
                    metavar=('START', 'COUNT'), \
                    help='range of time to apply deconvolution (COUNT = 0 for all)')

parser.add_argument('-f', '--use-fft', action = 'store_true', default = use_fft, \
                    help='Use FFT in the CPU mode')

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='Turn on GPU use with the specified ID')

parser.add_argument('-m', '--save-memory', action = 'store_true', default = save_memory, \
                    help='Save memory using float32 and complex64 (for GPU)')

parser.add_argument('input_file', default = input_filename, \
                    help='a multipage TIFF file to deconvolve')

args = parser.parse_args()

# defaults
time_range = args.time_range
iterations = args.number_of_iterations
psf_filename = args.psf_image
z_zoom_image = args.z_zoom_image
z_shrink_image = args.z_shrink_image
gpu_id = args.gpu_id
use_fft = args.use_fft
save_memory = args.save_memory

input_filename = args.input_file
if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filename) + output_suffix
    if output_filename == input_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file

# load psf image
if pathlib.Path(psf_filename).exists():
    print("Read PSF image in the current folder:", psf_filename)
    psf_image = tifffile.imread(psf_filename)
else:
    psf_path = pathlib.Path(psf_folder).joinpath(psf_filename)
    if psf_path.exists():
        print("Read PSF image in the system folder:", str(psf_path))
        psf_image = tifffile.imread(str(psf_path))
    else:
        raise Exception('PSF file {0} not found'.format(psf_filename))

# load input image
input_tiff = mmtiff.MMTiff(input_filename)
input_list = input_tiff.as_list()

# deconvolve
deconvolver = lucy.Lucy(psf_image, gpu_id, use_fft, save_memory)

time_start = time_range[0]
if time_range[1] == 0:
    time_count = len(input_list)
else:
    time_count = min(len(input_list), time_range[1])

if z_zoom_image and input_tiff.total_zstack > 1:
    zoom_ratio = input_tiff.z_step_um / input_tiff.pixelsize_um
    print("Setting Z-zoom:", zoom_ratio)
else:
    zoom_ratio = 1

# save results in the CTZYX order
output_list = []
print("Start deconvolution:", time.ctime())
for channel in range(input_tiff.total_channel):
    print("Prosessing channel {0}:".format(channel))
    output_frames = []
    print("Frames:", end = ' ')
    for index in range(time_start, time_count):
        input_image = input_list[index][:, channel]
        print(index, end = ' ', flush = True)
        output_frames.append(deconvolver.deconvolve(input_image, iterations, zoom_ratio, z_shrink_image))
    output_list.append(output_frames)
    print(".")
print("End deconvolution:", time.ctime())

# shape output into the TZCYX order
output_image = numpy.array(output_list)
output_axis = numpy.arange(output_image.ndim)
output_axis[0] = 1
output_axis[1] = 2
output_axis[2] = 0
output_image = output_image.transpose(output_axis)

if input_tiff.dtype.kind == 'i' or input_tiff.dtype.kind == 'u':
    output_image = mmtiff.MMTiff.float_to_int(output_image, input_tiff.dtype)

# output in the ImageJ format, dimensions should be in TZCYX order
input_tiff.save_image(output_filename, output_image)