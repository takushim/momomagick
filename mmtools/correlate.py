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
from scipy.ndimage.interpolation import shift, zoom
from scipy.signal import correlate

class Correlate:
    def __init__ (self):
        self.columns = ['align_plane', 'align_x', 'align_y', 'align_c']
        self.params = {'scale': 10, 'search_pixel': 20}
        self.invert_image = False

    def output_header (self, output_file, input_filename, reference_filename):
        output_file.write('## Alignment by TaniAuto (Auto-correlation) at %s\n' % (time.ctime()))
        output_file.write('#   file = \'%s\'; reference = %s\n' % (input_filename, reference_filename))
        output_file.write('#   scale = %d; search_pixel = %d\n' % (self.params['scale'], self.params['search_pixel']))

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

        reference_scaled = zoom(reference, self.params['scale'])
        for index in range(len(orig_images)):
            # scale image
            image_scaled = zoom(orig_images[index], self.params['scale'])
            print("Image scaled into:", image_scaled.shape)

            # prepare a table
            search_pairs = (2 * self.params['search_pixel'] + 1) ** 2
            search_array = numpy.arange(-self.params['search_pixel'], self.params['search_pixel'] + 1, 1)
            pairs = numpy.zeros(search_pairs, dtype=[('x', numpy.int), ('y', numpy.int), ('c', numpy.float)])

            # calculate autocorrelation
            for i, (x, y) in enumerate(itertools.product(search_array, search_array)):
                image_shifted = shift(image_scaled, (x, y))
                pairs[i]['x'] = x
                pairs[i]['y'] = y
                pairs[i]['c'] = numpy.sum(correlate(reference, image_shifted), dtype=numpy.float)
                print(index, x, y, pairs[i]['c'])

            # find the minimum
            index_min = numpy.amin(pairs['c'])
            move_x[index] = float(pairs[index_min]['x']) / self.params['scale']
            move_y[index] = float(pairs[index_min]['y']) / self.params['scale']
            move_c[index] = pairs[index_min]['c']
            print("Drift", index, move_x[index], move_y[index], move_c[index])

        # make pandas dataframe
        result = pandas.DataFrame({ \
                'align_plane' : numpy.arange(len(orig_images)), \
                'align_x' : move_x, \
                'align_y' : move_y, \
                'align_c' : move_c})

        return result[self.columns]
