#!/usr/bin/env python

import sys, argparse, numpy, math
from scipy.ndimage import rotate
from mmtools import mmtiff

# defaults
input_filename = None
filename_suffix = '_rotated.tif'
output_filename = None
#use_channel = 0
gpu_id = None

parser = argparse.ArgumentParser(description='Rotate a multipage TIFF image.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

#parser.add_argument('-c', '--use-channel', type=int, default=use_channel, \
#                    help='select one channel to be output')
#
parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='Turn on GPU use with the specified ID')

parser.add_argument('-s', '--save-memory', action = 'store_true', \
                    help='Save memory using float32 and complex64 (only for GPU)')

parser.add_argument('input_file', default = input_filename, \
                    help='input multpage-tiff file to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file
#use_channel = args.use_channel
gpu_id = args.gpu_id
save_memory = args.save_memory
if args.output_file is None:
    output_filename = mmtiff.MMTiff.stem(input_filename) + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file

# read input image(s) and expand it
input_tiff = mmtiff.MMTiff(input_filename)
#input_image = input_tiff.as_array(channel = use_channel, drop = True)
input_image = input_tiff.as_list()[0]

resized_shape = (numpy.array(input_image.shape) * math.sqrt(2)).astype(numpy.int)
resized_image = mmtiff.MMTiff.resize(input_image, resized_shape, center = True)

# load cupy if gpu_id is specified
if gpu_id is not None:
    import cupy
    from cupyx.scipy.ndimage import rotate as cupyx_rotate

    device = cupy.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))

    if save_memory:
        if resized_image.dtype.kind == 'i':
            gpu_resized_image = cupy.array(resized_image, dtype = numpy.int32)
        elif resized_image.dtype.kind == 'u':
            gpu_resized_image = cupy.array(resized_image, dtype = numpy.uint32)
        elif resized_image.dtype.kind == 'f':
            gpu_resized_image = cupy.array(resized_image, dtype = numpy.float32)
        else:
            raise Exception("Unsupported data type:", resized_image.dtype)
    else:
        print("Memory saving disabled. The GPU may give an out-of-memory error.")
        gpu_resized_image = cupy.array(resized_image)

# prepare an empty array
output_image_list = []
for angle in [0, 30]:
    print("Rotation:", angle)
    if gpu_id is None:
        rotated_image = rotate(resized_image, angle, axes = (1, 0), reshape = False)
        print(rotated_image.shape)
        output_image_list.append(rotated_image)
    else:
        gpu_rotated_image = cupyx_rotate(gpu_resized_image, angle, axes = (2, 1), resize = False)
        output_image_list.append(cupy.asnumpy(gpu_rotated_image))

# output multipage tiff, dimensions should be in TZCYX order
print("Output image file to %s." % (output_filename))
output_array = numpy.array(output_image_list)
print(output_array.shape)
#output_array = output_array[:, :, numpy.newaxis, :, :]
input_tiff.save_image(output_filename, output_array)
