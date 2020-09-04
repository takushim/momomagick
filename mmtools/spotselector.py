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

class SpotSelector:
    def __init__ (self):
        self.lifetime_min = 0
        self.lifetime_max = numpy.inf

    def filter_spots_lifetime (self, spot_table):
        spot_table = spot_table.sort_values(by = ['total_index', 'plane']).reset_index(drop=True)
        spot_table = spot_table[(self.lifetime_min <= spot_table.life_total) & \
                                (spot_table.life_total <= self.lifetime_max)].reset_index(drop=True)

        return spot_table

    def omit_lastplane_spots (self, spot_table, lastplane_index):
        total_indexes = spot_table[spot_table.plane == lastplane_index].total_index.tolist()
        total_indexes = list(set(total_indexes))

        return spot_table[~spot_table.total_index.isin(total_indexes)]

    def keep_first_spots (self, spot_table):
        spot_table = spot_table.drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)
        return spot_table

    def keep_last_spots (self, spot_table):
        spot_table = spot_table.drop_duplicates(subset='total_index', keep='last').reset_index(drop=True)
        return spot_table

    def average_spots (self, spot_table):
        agg_dict = {x : numpy.max for x in spot_table.columns}
        agg_dict['x'] = numpy.mean
        agg_dict['y'] = numpy.mean
        agg_dict['intensity'] = numpy.mean
        agg_dict['distance'] = numpy.sum
        spot_table = spot_table.groupby('total_index').agg(agg_dict).reset_index(drop=True)
        return spot_table

    def filter_spots_maskimage (self, spot_table, mask_image):
        first_spot_table = spot_table.drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)
        first_spot_table['mask'] = mask_image[first_spot_table.y.values.astype(numpy.int), first_spot_table.x.values.astype(numpy.int)]
        index_set = set(first_spot_table[first_spot_table['mask'] > 0].total_index.to_list())
        return spot_table[spot_table.total_index.isin(index_set)]


