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

import sys, numpy, pandas

class SpotPlotter:
    def __init__ (self):
        self.image_scale = 4
        self.align_each = 500

    def read_image_size (self, input_filename):
        params = self.read_image_params(input_filename)
        return int(params['width']), int(params['height'])

    def read_image_params (self, input_filename):
        params = {}
        with open(input_filename, 'r') as spot_file:
            for line in spot_file:
                if line.startswith('#') is False:
                    break
                line = line[1:].strip()
                exec(line, {}, params)

        return params

    def plot_spots (self, last_image, last_plane, spot_table, align_table):
        # prepare working array
        work_image = last_image.copy()

        # make spots dataframe
        spots = spot_table[['plane', 'x', 'y']].copy().reset_index(drop=True)

        # scale and alignment
        if align_table is not None:
            spots['align_index'] = ((spots['plane'] + last_plane) // self.align_each)
            spots = pandas.merge(spots, align_table, left_on='align_index', right_on='align_plane', how='left')
            spots['plot_x'] = ((spots['x']  - spots['align_x']) * self.image_scale).astype(numpy.int)
            spots['plot_y'] = ((spots['y']  - spots['align_y']) * self.image_scale).astype(numpy.int)
        else:
            spots['plot_x'] = (spots['x'] * self.image_scale).astype(numpy.int)
            spots['plot_y'] = (spots['y'] * self.image_scale).astype(numpy.int)

        # drop inappropriate spots
        height, width = work_image.shape
        spots = spots[(0 <= spots['plot_x']) & (spots['plot_x'] < width) & \
                      (0 <= spots['plot_y']) & (spots['plot_y'] < height)].reset_index(drop=True)

        # plot spots
        work_array = numpy.zeros(work_image.shape, dtype=numpy.int32)
        plot_x = spots.plot_x.values
        plot_y = spots.plot_y.values
        numpy.add.at(work_array, (plot_y, plot_x), 1)

        # combine with work_image
        work_image = work_image + work_array

        return work_image
