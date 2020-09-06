#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas
from mmtools import mmtiff, akaze

# prepare aligner
aligner = akaze.Akaze()

# defaults
input_filename = None
output_filename = 'align.txt'
ref_image_filename = None

parser = argparse.ArgumentParser(description='Calculate sample drift using A-KAZE feature matching', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--output-file', nargs=1, default = [output_filename], \
                    help='output TSV file name ({0} if not specified)'.format(output_filename))

parser.add_argument('-r', '--ref-image', nargs=1, default = [ref_image_filename], \
                    help='specify an image as a reference')

parser.add_argument('input_file', nargs=1, default = None, \
                    help='a multipage TIFF file(s) to align')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
output_filename = args.output_file[0]
ref_image_filename = args.ref_image[0]

# read input image
input_tiff = mmtiff.MMTiff(input_filename)
if input_tiff.colored:
    raise Exception('Input_image: color image not accepted')
if input_tiff.dtype != numpy.uint8:
    raise Exception('Input_image: image should be 8-bit') # FIXME

input_image = input_tiff.as_array()
input_image = input_image[:, 0, 0]
if input_tiff.total_zstack > 1 or input_tiff.total_channel > 1:
    print("Input image: zstack #0 and/or channel #0 is being used.")

# read reference image
if ref_image_filename is None:
    ref_tiff = None
    ref_image = None
else:
    ref_tiff = mmtiff.MMTiff(ref_image_filename)
    if ref_tiff.colored:
        raise Exception('Reference image: color reference image not accepted.')
    if ref_tiff.dtype != numpy.uint8:
        raise Exception('Reference image: image should be 8-bit') # FIXME

    ref_image = ref_tiff.as_array()
    ref_image = ref_image[0, 0, 0]
    if ref_image.total_time > 1 or ref_image.total_zstack > 1 or ref_image.total_channel > 1:
        print("Reference image: time #0 and/or zstack #0 and/or channel #0 is being used.")

# alignment
results = aligner.calculate_alignments(input_image, ref_image)

# open tsv file and write header
output_file = open(output_filename, 'w', newline='')
aligner.output_header(output_file, input_filename, ref_image_filename)
output_file.write('\t'.join(results.columns) + '\n')

# output result and close
results.to_csv(output_file, columns = results.columns, \
               sep='\t', index = False, header = False, mode = 'a')
output_file.close()
print("Output alignment tsv file to %s." % (output_filename))
