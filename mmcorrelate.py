#!/usr/bin/env python

import sys, argparse, pathlib, re, numpy, pandas, tifffile
from mmtools import correlate

# prepare spot marker
aligner = correlate.Correlate()

# default values
input_filename = None
output_filename = None
filename_suffix = '.txt'

# parse arguments
parser = argparse.ArgumentParser(description='Calculate sample drift using correlation', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output-file', nargs=1, default=output_filename, \
                    help='output txv file ([basename]%s by default)' % (filename_suffix))

parser.add_argument('input_file', nargs=1, default=input_filename, \
                    help='input TIFF file to plot spots')
args = parser.parse_args()

# set arguments
input_filename = args.input_file[0]
if args.output_file is None:
    stem = pathlib.Path(input_filename).stem
    stem = re.sub('\.ome$', '', stem, flags=re.IGNORECASE)
    output_filename = stem + filename_suffix
    if input_filename == output_filename:
        raise Exception('input_filename == output_filename.')
else:
    output_filename = args.output_file[0]

# read TIFF files
input_images = tifffile.imread(input_filename)

# alignment
results = aligner.calculate_alignments(input_images, None)

# open tsv file and write header
output_tsv_file = open(output_filename, 'w', newline='')
aligner.output_header(output_file, input_filenames[0], None)
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

