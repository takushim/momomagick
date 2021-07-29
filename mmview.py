#!/usr/bin/env python

import sys, argparse, numpy, math
from scipy.ndimage import rotate, zoom
from mmtools import mmtiff, gpuimage

# defaults
input_filename = None
filename_suffix = '_view.tif'
output_filename = None
gpu_id = None
tilt_angle = 45
tilt_axis = 'x'
rotation_range = [0, 360, 10]

parser = argparse.ArgumentParser(description='Obtain diSPIM view for a given image.', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-file', default = output_filename, \
                    help='output image file name ([basename]{0} by default)'.format(filename_suffix))

parser.add_argument('-g', '--gpu-id', default = gpu_id, \
                    help='Turn on GPU use with the specified ID')

parser.add_argument('-s', '--save-memory', action = 'store_true', \
                    help='Save memory using float32 and complex64 (only for GPU)')

parser.add_argument('-t', '--tilt-angle', type=float, default = tilt_angle, \
                    help='Angle of tilting to obtain the diSPIM plane')

parser.add_argument('-a', '--tilt-axis', default = tilt_axis, type = str.lower, \
                    choices = ['x', 'y'], \
                    help='Axis to rotate the image')

parser.add_argument('-r', '--rotation-range', nargs=3, type=float, \
                    default = rotation_range, \
                    metavar=('BEGIN', 'END', 'STEP'), \
                    help='Range of rotation angles')

parser.add_argument('input_file', default = input_filename, \
                    help='input multpage-tiff file')

args = parser.parse_args()

# set arguments
input_filename = args.input_file
gpu_id = args.gpu_id
save_memory = args.save_memory
tilt_axis = args.tilt_axis
tilt_angle = args.tilt_angle
rotation_range = args.rotation_range

if args.output_file is None:
    output_filename = mmtiff.stem(input_filename) + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename')
else:
    output_filename = args.output_file

# read input image(s) and expand it
input_tiff = mmtiff.MMTiff(input_filename)
if input_tiff.total_time > 1:
    print("Using time = 0 only.")
input_image = input_tiff.as_list()[0]

# load cupy if gpu_id is specified
if gpu_id is not None:
    import cupy
    from cupyx.scipy.ndimage import rotate as cupyx_rotate
    from cupyx.scipy.ndimage import zoom as cupyx_zoom

    device = cupy.cuda.Device(gpu_id)
    device.use()
    print("Turning on GPU: {0}, PCI-bus ID: {1}".format(gpu_id, device.pci_bus_id))
    print("Free memory:", device.mem_info)

# set size and rotation tuple
shape = numpy.array(input_image.shape).astype(numpy.float)
z_ratio = input_tiff.z_step_um / input_tiff.pixelsize_um
diag = math.sqrt((shape[0] * z_ratio)**2 + shape[2]**2 + shape[3]**2)

resized_shape = numpy.array([diag, diag, diag]).astype(int)
print("Resized shape:", resized_shape)

if tilt_axis == 'x':
    tilt_tuple = (0, 1)
elif tilt_axis == 'y':
    tilt_tuple = (0, 2)
else:
    raise Exception('Unknown axis:', tilt_axis)

output_image_list_all = []

for channel in range(input_tiff.total_channel):
    print("Channel:", channel)
    image = input_image[:, channel]
    zoom_tuple = [z_ratio, 1, 1]

    if gpu_id is None:
        zoomed_image = zoom(image, zoom = zoom_tuple)
        resized_image = gpuimage.resize(zoomed_image, resized_shape, center = True)
    else:
        device = cupy.cuda.Device(gpu_id)
        mempool = cupy.get_default_memory_pool()
        if save_memory:
            if input_image.dtype.alignment > 4:
                if input_image.dtype.kind == 'i':
                    data_type = numpy.int32
                elif input_image.dtype.kind == 'u':
                    data_type = numpy.uint32
                elif input_image.dtype.kind == 'f':
                    data_type = numpy.float32                
                else:
                    raise Exception("Unsupported data type:", input_image.dtype)
            else:
                data_type = input_image.dtype
        else:
            print("Memory saving disabled. The GPU may give an out-of-memory error.")
            data_type = input_image.dtype
        
        print("Using data type:", data_type.name)
        gpu_input_image = cupy.array(image, dtype = data_type)
        gpu_zoomed_image = cupyx_zoom(gpu_input_image, zoom = zoom_tuple, order = 1)

        gpu_resized_image = cupy.zeros(resized_shape, dtype = data_type)
        slices_source, slices_target = \
            mmtiff.paste_slices(gpu_zoomed_image.shape, gpu_resized_image.shape, \
                                center = True)
        gpu_resized_image[slices_target] = gpu_zoomed_image[slices_source].copy()
        print("Memory: {0} vs {1}".format(mempool.used_bytes(), mempool.free_bytes()))
        print("Free memory:", device.mem_info)

        gpu_input_image = None
        gpu_zoomed_image = None
        print("Memory: {0} vs {1}".format(mempool.used_bytes(), mempool.free_bytes()))
        print("Free memory:", device.mem_info)

    # prepare an empty array
    output_image_list = []
    print("Rotation:", end = ' ')
    rotate_tuple = (1, 2)
    for angle in range(*rotation_range):
        print(angle, end = ' ', flush = True)
        if gpu_id is None:
            rotated_image = rotate(resized_image, angle, axes = rotate_tuple, reshape = False)
            rotated_image = rotate(rotated_image, tilt_angle, axes = tilt_tuple, reshape = False)
        else:
            gpu_rotated_image = cupyx_rotate(gpu_resized_image, angle, \
                                             axes = rotate_tuple, order = 1, reshape = False)
            gpu_rotated_image = cupyx_rotate(gpu_rotated_image, tilt_angle, \
                                             axes = tilt_tuple, order = 1, reshape = False)
            rotated_image = cupy.asnumpy(gpu_rotated_image)
        output_image_list.append(rotated_image)
    print(".")

output_image_list_all.append(output_image_list)

# output multipage tiff, dimensions should be in TZCYX order
print("Output image file to %s." % (output_filename))
output_array = numpy.array(output_image_list_all)
print(output_array.shape)
output_array = output_array.transpose((1, 2, 0, 3, 4))
print(output_array.shape)

output_array = output_array.astype(input_tiff.dtype)
input_tiff.save_image_ome(output_filename, output_array)
