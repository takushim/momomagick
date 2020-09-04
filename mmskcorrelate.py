#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas, tifffile
from mmtools import skcorrelate
from scipy.ndimage.interpolation import shift, zoom

# prepare spot marker
aligner = skcorrelate.SkCorrelate()

# default values
input_filename = None
output_tsv_filename = 'align.txt'
output_image = False
output_image_scale = 1
output_image_filename = None
filename_suffix = '_skcorr.tif'
#reference_image_filename = None

# parse arguments
parser = argparse.ArgumentParser(description='Calculate sample drift using correlation', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--output-tsv-file', nargs=1, default = [output_tsv_filename], \
                    help='output TSV file name (%s if not specified)' % (output_tsv_filename))

#parser.add_argument('-r', '--reference-image', nargs=1, default = [reference_image_filename], \
#                    help='use an external reference image')

parser.add_argument('-O', '--output-image', action='store_true', default=output_image, \
                    help='output image after drift correction')
parser.add_argument('-o', '--output-image-file', nargs=1, default = output_image_filename, \
                    help='output image file name ([basename]%s if not specified)' % (filename_suffix))
parser.add_argument('-x', '--output-image-scale', nargs=1, type=int, default=[output_image_scale], \
                    help='scale of output image')

parser.add_argument('-i', '--invert-image', action='store_true', default=aligner.invert_image, \
                    help='invert the LUT of output image')

parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input TIFF file to plot spots')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
aligner.invert_image = args.invert_image
output_tsv_filename = args.output_tsv_file[0]
#reference_image_filename = args.reference_image[0]
output_image = args.output_image
output_image_filename = args.output_image_file
output_image_scale = args.output_image_scale[0]
if args.output_image_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_image_filename = stem + filename_suffix
    if input_filename == output_image_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_image_filename = args.output_image_file[0]

# read TIFF files
input_images = tifffile.imread(input_filename)

# alignment
results = aligner.calculate_alignments(input_images, None)

# open tsv file and write header
output_tsv_file = open(output_tsv_filename, 'w', newline='')
aligner.output_header(output_tsv_file, input_filename, None)
output_tsv_file.write('\t'.join(results.columns) + '\n')

# output result and close
results.to_csv(output_tsv_file, columns = results.columns, \
               sep='\t', index = False, header = False, mode = 'a')
output_tsv_file.close()
print("Output alignment tsv file to %s." % (output_tsv_filename))

# output ImageJ, dimensions should be in TZCYXS order
#print('Output image was shaped into:', output_image.shape)
#tifffile.imsave(output_filename, output_image, imagej = True, \
#                resolution = (1 / xy_resolution, 1 / xy_resolution), \
#                metadata = {'spacing': z_spacing, 'unit': 'um', 'Composite mode': 'composite'})

# output image
if output_image is True:
    images_uint8 = aligner.convert_to_uint8(input_images)
    output_image_list = []
    for row, align in results.iterrows():
        plane = results.align_plane[row]
        if plane not in range(len(images_uint8)):
            print("Skip plane %d due to out-of-range." % (results.plane[row]))
            continue
        output_image = zoom(images_uint8[plane], output_image_scale)
        output_image = shift(output_image, (int(-align.align_x * output_image_scale), int(-align.align_y * output_image_scale)))
        output_image_list.append(output_image)

    # output multipage tiff
    print("Output image file to %s." % (output_image_filename))
    tifffile.imsave(output_image_filename, numpy.array(output_image_list))
