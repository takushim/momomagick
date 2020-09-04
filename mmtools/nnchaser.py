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

import sys, numpy, pandas, time
from sklearn.neighbors import NearestNeighbors

class NNChaser:
    def __init__ (self):
        self.chase_distance = 3.0

    def output_header (self, output_file):
        output_file.write('## Chased by TaniChaser at %s\n' % (time.ctime()))
        output_file.write('#   chase_distance = %f\n' % (self.chase_distance))

    def chase_spots (self, spot_table):
        numpy.set_printoptions(threshold=numpy.inf)
        results = []

        for index in range(max(spot_table.plane)):
            # get partial spot table
            orig_spots = spot_table[spot_table.plane == index]
            next_spots = spot_table[spot_table.plane == index + 1]
            if len(orig_spots) == 0:
                results.append(numpy.array([]))
                continue

            pairs = numpy.zeros(len(orig_spots), dtype=[('orig_array_index', numpy.int), \
                                                        ('orig_total_index', numpy.int), \
                                                        ('next_array_index', numpy.int), \
                                                        ('distance', numpy.float), \
                                                        ('track_distance', numpy.float), \
                                                        ('valid', numpy.bool)])

            if len(next_spots) == 0:
                # do not run nn since there are no spots on the next plane
                pairs['orig_array_index'] = numpy.arange(len(orig_spots))
                pairs['orig_total_index'] = orig_spots.total_index.values
                pairs['next_array_index'] = -1
                pairs['distance'] = 0.0
                pairs['track_distance'] = 0.0
                pairs['valid'] = False
            else:
                # nearest neighbor to find nearest spots
                nn = NearestNeighbors(n_neighbors = 1, metric = 'euclidean').fit(next_spots[['x', 'y']].values)
                distances, targets = nn.kneighbors(orig_spots[['x', 'y']].values)

                # make numpy array to find duplicates and too far spots
                pairs['orig_array_index'] = numpy.arange(len(orig_spots))
                pairs['orig_total_index'] = orig_spots.total_index.values
                pairs['next_array_index'] = targets.flatten()
                pairs['distance'] = distances.flatten()
                pairs['track_distance'] = 0.0
                pairs['valid'] = True

                # omit too far spots
                pairs['valid'][pairs['distance'] > self.chase_distance] = False

                # find duplicated targets
                pairs = numpy.sort(pairs, order=['next_array_index', 'distance'])
                unique_index = numpy.unique(pairs['next_array_index'], return_index = True)[1]

                # omit duplicates
                mask = numpy.ones(len(pairs), dtype=numpy.bool)
                mask[unique_index] = False
                pairs['valid'][mask] = False

                # delete next_index and distance
                pairs['next_array_index'][pairs['valid'] == False] = -1
                pairs['distance'][pairs['valid'] == False] = 0.0

            # sort again and save
            pairs = numpy.sort(pairs, order=['orig_array_index'])
            results.append(pairs)

        # add last results
        lastplane_spots = spot_table[spot_table.plane == max(spot_table.plane)]
        pairs = numpy.zeros(len(lastplane_spots), dtype=[('orig_array_index', numpy.int), \
                                                         ('orig_total_index', numpy.int), \
                                                         ('next_array_index', numpy.int), \
                                                         ('distance', numpy.float), \
                                                         ('track_distance', numpy.float), \
                                                         ('valid', numpy.bool)])
        pairs['orig_array_index'] = numpy.arange(len(lastplane_spots))
        pairs['orig_total_index'] = lastplane_spots.total_index.values
        pairs['next_array_index'] = -1
        pairs['distance'] = 0.0
        pairs['track_distance'] = 0.0
        pairs['valid'] = False
        results.append(pairs)

        # chase spots by updating total_index
        for index in range(max(spot_table.plane)):
            orig_pairs = results[index]
            next_pairs = results[index + 1]

            if (len(orig_pairs) == 0) or (len(next_pairs) == 0):
                continue

            orig_indexes = orig_pairs['orig_array_index'][orig_pairs['valid'] == True]
            next_indexes = orig_pairs['next_array_index'][orig_indexes]
            next_pairs['orig_total_index'][next_indexes] = orig_pairs['orig_total_index'][orig_indexes]
            next_pairs['track_distance'][next_indexes] = orig_pairs['distance'][orig_indexes]

        # update indexes and make a distance column
        spot_table['total_index'] = numpy.concatenate([result['orig_total_index'] for result in results if len(result) > 0])
        spot_table['distance'] = numpy.concatenate([result['track_distance'] for result in results if len(result) > 0])

        # sort table
        spot_table = spot_table.sort_values(by = ['total_index', 'plane']).reset_index(drop=True)

        # add life column
        spot_table['life_index'] = (spot_table.groupby('total_index').cumcount())
        lifetime_table = spot_table['total_index'].value_counts().to_frame('life_total')
        spot_table = pandas.merge(spot_table, lifetime_table, \
                                    left_on='total_index', right_index=True, how='left')

        return spot_table
