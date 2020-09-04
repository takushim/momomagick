#!/usr/bin/env python

# Copyright (c) 2018-2019, Takushi Miyoshi
# Copyright (c) 2012-2019, Department of Otolaryngology, 
#                          Graduate School of Medicine, Kyoto University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys, numpy, pandas, time, itertools
from skimage.registration import phase_cross_correlation

class SkCorrelate:
    def __init__ (self):
        self.columns = ['align_plane', 'align_x', 'align_y', 'align_c']
        self.params = {'scale': 10}
        self.invert_image = False

    def output_header (self, output_file, input_filename, reference_filename):
        output_file.write('## Alignment by SkCorrelation at %s\n' % (time.ctime()))
        output_file.write('#   file = \'%s\'; reference = %s\n' % (input_filename, reference_filename))
        output_file.write('#   scale = %d\n' % (self.params['scale']))

    def convert_to_uint8 (self, orig_image):
        images_uint8 = numpy.zeros(orig_image.shape, dtype = numpy.uint8)

        image_type = orig_image.dtype.name
        if image_type == 'int32' or image_type == 'uint32' or image_type == 'uint16':
            mean = numpy.mean(orig_image)
            sigma = numpy.std(orig_image)
            image_min = max(0, mean - 3 * sigma)
            image_max = min(mean + 4 * sigma, numpy.iinfo(orig_image.dtype).max)
            images_uint8 = (255.0 * (orig_image - image_min) / (image_max - image_min)).clip(0, 255).astype(numpy.uint8)
        elif image_type == 'uint8':
            images_uint8 = orig_image
        else:
            raise Exception('invalid image file format')

        if self.invert_image is True:
            image_color = 255 - image_color

        return images_uint8

    def calculate_alignments (self, orig_images, reference = None):
        # array for the results
        move_x = numpy.zeros(len(orig_images))
        move_y = numpy.zeros(len(orig_images))
        move_c = numpy.zeros(len(orig_images))

        # params of original image
        if reference is None:
            reference = orig_images[0]

        for index in range(len(orig_images)):
            shifts, error, phasediff = phase_cross_correlation(reference, orig_images[index], space = 'real', upsample_factor=self.params['scale'])
            move_y[index], move_x[index] = shifts
            move_c[index] = error
            print("Drift", move_x[index], move_y[index], move_c[index])

        # make pandas dataframe
        result = pandas.DataFrame({ \
                'align_plane' : numpy.arange(len(orig_images)), \
                'align_x' : move_x, \
                'align_y' : move_y, \
                'align_c' : move_c})

        return result[self.columns]
