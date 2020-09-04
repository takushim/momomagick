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

import os, sys, numpy, pandas, time
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
from sklearn.neighbors import NearestNeighbors

class Gaussian8:
    def __init__ (self):
        self.laplace = 2.0 # Diameter of Spots
        self.min_distance = 1 # Pixel area (int) to find local max (usually 1)
        self.threshold_abs = 0.006 # Threshold to find local max
        self.max_diameter = 10.0
        self.dup_threshold = 3.0
        self.columns = ['total_index', 'plane', 'index', 'x', 'y', 'diameter', 'intensity', 'fit_error', 'chi_square']
        self.image_clip_min = 0.0
        self.image_clip_max = numpy.iinfo(numpy.int32).max

    def output_header (self, output_file, input_filename, image_array):
        filename = os.path.basename(input_filename)
        planes = image_array.shape[0]
        if len(image_array.shape) == 2:
            planes = 1

        #params = {'input_file': filename, 'total_planes': planes, \
        #          'width': image_array.shape[2], 'height': image_array.shape[1], \
        #          'laplace': self.laplace. 'min_distance': self.min_distance, \
        #          'threshold_abs': self.threshold_abs, \
        #          'image_clip_min': self.image_clip_min, 'image_clip_max': self.image_clip_max}

        output_file.write('## Traced by TaniTracer at %s for %s\n' % (time.ctime(), filename))
        output_file.write('#   total_planes = %d; width = %d; height = %d\n' %\
                          (planes, image_array.shape[2], image_array.shape[1]))
        output_file.write('#   laplace = %f; min_distance = %d; threshold_abs = %f\n' %\
                          (self.laplace, self.min_distance, self.threshold_abs))
        output_file.write('#   max_diameter = %f; dup_threshold = %f\n' %\
                          (self.max_diameter, self.dup_threshold))
        output_file.write('#   image_clip_min = %f; image_clip_max = %f\n' %\
                          (self.image_clip_min, self.image_clip_max))

    def set_image_clip (self, image_array):
        self.image_clip_min = numpy.percentile(image_array, 0.1)
        self.image_clip_max = numpy.percentile(image_array, 99.9)

    def gaussian_fitting (self, input_image, float_image):
        # Find local max at 1-pixel resolution (order: [y, x])
        xy = peak_local_max(float_image, min_distance = self.min_distance,\
                            threshold_abs = self.threshold_abs, exclude_border = True)

        # Calculate subpixel correction (x = xy[:,1], y = xy[:,0])
        c10 = ( - numpy.log(float_image[xy[:,0] - 1, xy[:,1] - 1]) - numpy.log(float_image[xy[:,0], xy[:,1] - 1]) \
                - numpy.log(float_image[xy[:,0] + 1, xy[:,1] - 1]) + numpy.log(float_image[xy[:,0] - 1, xy[:,1] + 1]) \
                + numpy.log(float_image[xy[:,0], xy[:,1] + 1]) + numpy.log(float_image[xy[:,0] + 1, xy[:,1] + 1]) ) / 6
        c01 = ( - numpy.log(float_image[xy[:,0] - 1, xy[:,1] - 1]) - numpy.log(float_image[xy[:,0] - 1, xy[:,1]]) \
                - numpy.log(float_image[xy[:,0] - 1, xy[:,1] + 1]) + numpy.log(float_image[xy[:,0] + 1, xy[:,1] - 1]) \
                + numpy.log(float_image[xy[:,0] + 1, xy[:,1]]) + numpy.log(float_image[xy[:,0] + 1, xy[:,1] + 1]) ) / 6
        c20 = (   numpy.log(float_image[xy[:,0] - 1, xy[:,1] - 1]) + numpy.log(float_image[xy[:,0], xy[:,1] - 1]) \
                + numpy.log(float_image[xy[:,0] + 1, xy[:,1] - 1]) - 2 * numpy.log(float_image[xy[:,0] - 1,xy[:,1]]) \
                - 2 * numpy.log(float_image[xy[:,0], xy[:,1]]) - 2 * numpy.log(float_image[xy[:,0] + 1, xy[:,1]]) \
                + numpy.log(float_image[xy[:,0] - 1, xy[:,1] + 1]) + numpy.log(float_image[xy[:,0], xy[:,1] + 1]) \
                + numpy.log(float_image[xy[:,0] + 1, xy[:,1] + 1]) ) / 6
        c02 = (   numpy.log(float_image[xy[:,0] - 1, xy[:,1] - 1]) + numpy.log(float_image[xy[:,0] - 1,xy[:,1]]) \
                + numpy.log(float_image[xy[:,0] - 1, xy[:,1] + 1]) - 2 * numpy.log(float_image[xy[:,0], xy[:,1] - 1]) \
                - 2 * numpy.log(float_image[xy[:,0], xy[:,1]]) - 2 * numpy.log(float_image[xy[:,0], xy[:,1] + 1]) \
                + numpy.log(float_image[xy[:,0] + 1, xy[:,1] - 1]) + numpy.log(float_image[xy[:,0] + 1,xy[:,1]]) \
                + numpy.log(float_image[xy[:,0] + 1, xy[:,1] + 1]) ) / 6
        c00 = ( - numpy.log(float_image[xy[:,0] - 1, xy[:,1] - 1]) + 2 * numpy.log(float_image[xy[:,0], xy[:,1] - 1]) \
                - numpy.log(float_image[xy[:,0] + 1, xy[:,1] - 1]) + 2 * numpy.log(float_image[xy[:,0] - 1,xy[:,1]]) \
                + 5 * numpy.log(float_image[xy[:,0], xy[:,1]]) + 2 * numpy.log(float_image[xy[:,0] + 1, xy[:,1]]) \
                - numpy.log(float_image[xy[:,0] - 1, xy[:,1] + 1]) + 2 * numpy.log(float_image[xy[:,0], xy[:,1] + 1]) \
                - numpy.log(float_image[xy[:,0] + 1, xy[:,1] + 1]) ) / 9

        fit_error = ( c00 - c10 + c20 - c01 + c02 - numpy.log(float_image[xy[:,0] - 1, xy[:,1] - 1]) )**2 \
            + ( c00 - c10 + c20 - numpy.log(float_image[xy[:,0], xy[:,1] - 1]) )**2 \
            + ( c00 - c10 + c20 + c01 + c02 - numpy.log(float_image[xy[:,0] + 1, xy[:,1] - 1]) )**2 \
            + ( c00 - c01 + c02 - numpy.log(float_image[xy[:,0] - 1, xy[:,1]]) )**2 \
            + ( c00 - numpy.log(float_image[xy[:,0], xy[:,1]]) )**2 \
            + ( c00 + c01 + c02 - numpy.log(float_image[xy[:,0] + 1, xy[:,1]]) )**2 \
            + ( c00 + c10 + c20 - c01 + c02 - numpy.log(float_image[xy[:,0] - 1, xy[:,1] + 1]) )**2 \
            + ( c00 + c10 + c20 - numpy.log(float_image[xy[:,0], xy[:,1] + 1]) )**2 \
            + ( c00 + c10 + c20 + c01 + c02 - numpy.log(float_image[xy[:,0] + 1, xy[:,1] + 1]) )**2

        chi_square = ( c00 - c10 + c20 - c01 + c02 - numpy.log(float_image[xy[:,0] - 1, xy[:,1] - 1]) )**2 / numpy.abs(numpy.log(float_image[xy[:,0] - 1, xy[:,1] - 1])) \
                   + ( c00 - c10 + c20 - numpy.log(float_image[xy[:,0], xy[:,1] - 1]) )**2 / numpy.abs(numpy.log(float_image[xy[:,0], xy[:,1] - 1])) \
                   + ( c00 - c10 + c20 + c01 + c02 - numpy.log(float_image[xy[:,0] + 1, xy[:,1] - 1]) )**2  / numpy.abs(numpy.log(float_image[xy[:,0] + 1, xy[:,1] - 1])) \
                   + ( c00 - c01 + c02 - numpy.log(float_image[xy[:,0] - 1, xy[:,1]]) )**2  / numpy.abs(numpy.log(float_image[xy[:,0] - 1, xy[:,1]])) \
                   + ( c00 - numpy.log(float_image[xy[:,0], xy[:,1]]) )**2  / numpy.abs(numpy.log(float_image[xy[:,0], xy[:,1]])) \
                   + ( c00 + c01 + c02 - numpy.log(float_image[xy[:,0] + 1, xy[:,1]]) )**2  / numpy.abs(numpy.log(float_image[xy[:,0] + 1, xy[:,1]])) \
                   + ( c00 + c10 + c20 - c01 + c02 - numpy.log(float_image[xy[:,0] - 1, xy[:,1] + 1]) )**2  / numpy.abs(numpy.log(float_image[xy[:,0] - 1, xy[:,1] + 1])) \
                   + ( c00 + c10 + c20 - numpy.log(float_image[xy[:,0], xy[:,1] + 1]) )**2  / numpy.abs(numpy.log(float_image[xy[:,0], xy[:,1] + 1])) \
                   + ( c00 + c10 + c20 + c01 + c02 - numpy.log(float_image[xy[:,0] + 1, xy[:,1] + 1]) )**2  / numpy.abs(numpy.log(float_image[xy[:,0] + 1, xy[:,1] + 1]))

        x = xy[:,1] - 0.5 * (c10/c20)
        y = xy[:,0] - 0.5 * (c01/c02)
        diameter = 2 * numpy.sqrt(- (0.5/c20 + 0.5/c02) / 2)
        intensity = input_image[xy[:,0], xy[:,1]]

        # make result dictionary
        result_dict = {'x': x, 'y': y, 'fit_error': fit_error, 'chi_square': chi_square, 'diameter': diameter, 'intensity': intensity}
        error_dict = {}

        # omit spots of abnormal subpixel correction (this should be run first of all)
        indexes = numpy.ones(len(result_dict['x']), dtype=numpy.bool)
        indexes = indexes & ((0.5 * (c10/c20)) < 1)
        indexes = indexes & ((0.5 * (c01/c02)) < 1)
        error_dict['large_subpixel_shift'] = len(result_dict['x']) - numpy.sum(indexes)
        result_dict = {k: result_dict[k][indexes] for k in result_dict}

        # omit nan spots
        indexes = numpy.ones(len(result_dict['x']), dtype=numpy.bool)
        indexes = indexes & (result_dict['x'] >= 0) & (result_dict['x'] <= float_image.shape[1])
        indexes = indexes & (result_dict['y'] >= 0) & (result_dict['y'] <= float_image.shape[0])
        error_dict['nan_coordinate'] = len(result_dict['x']) - numpy.sum(indexes)
        result_dict = {k: result_dict[k][indexes] for k in result_dict}

        # omit spots of large diameter
        indexes = numpy.ones(len(result_dict['x']), dtype=numpy.bool)
        indexes = indexes & (result_dict['diameter'] <= self.max_diameter)
        error_dict['large_diameter'] = len(result_dict['x']) - numpy.sum(indexes)
        result_dict = {k: result_dict[k][indexes] for k in result_dict}

        # omit duplicated spots
        if len(result_dict['x']) > 1:
            indexes = numpy.ones(len(result_dict['x']), dtype=numpy.bool)

            # find nearest spots
            nn = NearestNeighbors(n_neighbors = 2, metric = 'euclidean').fit(numpy.array([result_dict['x'], result_dict['y']]).T)
            distances, targets = nn.kneighbors(numpy.array([result_dict['x'], result_dict['y']]).T)
            distances, targets = distances[:,1], targets[:,1]
            pairs = numpy.zeros(len(result_dict['x']), dtype=[('orig_index', numpy.int), \
                                                              ('near_index', numpy.int), \
                                                              ('distance', numpy.float), \
                                                              ('fit_error', numpy.float), \
                                                              ('duplicated', numpy.bool)])
            pairs['orig_index'] = numpy.arange(len(result_dict['x']))
            pairs['near_index'] = targets
            pairs['distance'] = distances
            pairs['fit_error'] = result_dict['fit_error']
            pairs['duplicated'] = False

            # find duplicated points
            for pair in pairs:
                if (pair['distance'] <= self.dup_threshold) and (pairs[pair['near_index']]['near_index'] == pair['orig_index']):
                    if pair['fit_error'] > pairs[pair['near_index']]['fit_error']:
                        pairs[pair['orig_index']]['duplicated'] = True
                    else:
                        pairs[pair['near_index']]['duplicated'] = True

            # update result_dict
            indexes = (pairs['duplicated'] == False)
            error_dict['duplicated'] = len(result_dict['x']) - numpy.sum(indexes)
            result_dict = {k: result_dict[k][indexes] for k in result_dict}
        else:
            error_dict['duplicated'] = 0

        return result_dict, error_dict

    def clip_array (self, float_array):
        return float_array.clip(self.image_clip_min, self.image_clip_max)

    def standardize_and_filter_image (self, float_image):
        float_image = - (float_image - numpy.max(float_image)) / numpy.ptp(float_image)
        return ndimage.gaussian_laplace(float_image, self.laplace)

    def convert_to_pandas (self, result):
        length = max([len(item) for item in result.values()])
        result.update({'total_index' : numpy.arange(length)})
        return pandas.DataFrame(result, columns = self.columns)

    def fitting_image_array (self, input_image):
        numpy.seterr(divide='ignore', invalid='ignore')

        # get float image anf filter
        float_image = numpy.array(input_image, 'f')
        float_image = self.clip_array(float_image)
        float_image = self.standardize_and_filter_image(float_image)

        # fitting
        result, error = self.gaussian_fitting(input_image, float_image)

        # report error
        print("Dropped spots: %s" % (str(error)))

        # Make Pandas dataframe
        length = max([len(item) for item in result.values()])
        result.update({'plane': numpy.full(length, 0), 'index': numpy.arange(length)})
        spot_table = self.convert_to_pandas(result)

        return spot_table

    def fitting_image_stack (self, input_stack):
        numpy.seterr(divide='ignore', invalid='ignore')

        # get float image anf filter
        float_stack = numpy.array(input_stack, 'f')
        float_stack = self.clip_array(float_stack)

        # arrays to store results
        result_array = []
        error_array = []

        for index in range(len(input_stack)):
            # filter and fitting
            float_stack[index] = self.standardize_and_filter_image(float_stack[index])
            result, error = self.gaussian_fitting(input_stack[index], float_stack[index])

            # add plane and index
            length = max([len(item) for item in result.values()])
            result.update({'plane': numpy.full(length, index), 'index': numpy.arange(length)})

            # append to arrays
            result_array.append(result)
            error_array.append(error)

        # accumulate result
        result_concat = {}
        for key in result_array[0].keys():
            result_concat[key] = numpy.concatenate([result[key] for result in result_array])

        # sum error spots
        error_sum = {}
        for key in error_array[0].keys():
            error_sum[key] = numpy.sum([error[key] for error in error_array])
        print("Dropped spots: %s" % (str(error_sum)))

        # make pandas table
        spot_table = self.convert_to_pandas(result_concat)
        spot_table['total_index'] = numpy.arange(len(spot_table))
        return spot_table
