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

import sys, pandas, numpy
from PIL import Image, ImageDraw

class SpotMarker:
    def __init__ (self):
        self.marker_size = 4
        self.marker_width = 1
        self.mark_regression = False
        self.force_mark_emerge = False
        self.invert_image = False
        self.marker_colors = ['red', 'orange', 'blue', 'cyan']
        self.marker_rainbow = False
        self.rainbow_colors = numpy.array(["red", "blue", "green", "magenta", "purple", "cyan",\
                                           "orange", "maroon"])

    def convert_to_color (self, orig_image):
        if (len(orig_image.shape) > 3):
            print("Image with dimension > 3 was passed to conversion to color. Might be colored image. Conversion ignored.")
            return orig_image

        image_color = numpy.zeros(orig_image.shape + (3,), dtype = numpy.uint8)

        image_type = orig_image.dtype.name
        if image_type == 'int32' or image_type == 'uint16':
            mean = numpy.mean(orig_image)
            sigma = numpy.std(orig_image)
            image_min = max(0, mean - 3 * sigma)
            image_max = min(mean + 4 * sigma, numpy.iinfo(orig_image.dtype).max)
            image_8bit = (255.0 * (orig_image - image_min) / (image_max - image_min)).clip(0, 255).astype(numpy.uint8)
            image_color[:,:,:,0] = image_color[:,:,:,1] = image_color[:,:,:,2] = image_8bit
        elif image_type == 'uint8':
            image_color[:,:,:,0] = image_color[:,:,:,1] = image_color[:,:,:,2] = orig_image
        else:
            raise Exception('invalid image file format')
        
        if self.invert_image is True:
            image_color = 255 - image_color

        return image_color

    def convert_image_to_color (self, orig_image):
        if (len(orig_image.shape) > 3):
            print("Image with dimension > 3 was passed to conversion to color. Might be colored image. Conversion ignored.")
            return orig_image

        image_color = numpy.zeros(orig_image.shape + (3,), dtype = numpy.uint8)

        image_type = orig_image.dtype.name
        if image_type == 'int32' or image_type == 'uint16':
            mean = numpy.mean(orig_image)
            sigma = numpy.std(orig_image)
            image_min = max(0, mean - 3 * sigma)
            image_max = min(mean + 4 * sigma, numpy.iinfo(orig_image.dtype).max)
            image_8bit = (255.0 * (orig_image - image_min) / (image_max - image_min)).clip(0, 255).astype(numpy.uint8)
            image_color[:,:,0] = image_color[:,:,1] = image_color[:,:,2] = image_8bit
        elif image_type == 'uint8':
            image_color[:,:,0] = image_color[:,:,1] = image_color[:,:,2] = orig_image
        else:
            raise Exception('invalid image file format')
        
        if self.invert_image is True:
            image_color = 255 - image_color

        return image_color

    def tracking_status (self, spot_table):
        total_indexes = spot_table.total_index.tolist()

        status = ['cont' for i in range(len(total_indexes))]
        status[0] = 'new'

        for i in range(len(status) - 1):
            if total_indexes[i] < total_indexes[i + 1]:
                #print(total_indexes[i], total_indexes[i+1])
                if status[i] == 'new':
                    status[i], status[i + 1] = 'one', 'new'
                else:
                    status[i], status[i + 1] = 'end', 'new'
        if status[-1] == 'new':
            status[-1] = 'one'
        else:
            status[-1] = 'end'

        return status

    def mark_spots (self, image_color, spot_table):
        # copy for working
        work_table = spot_table.copy()

        # set colors
        if self.marker_rainbow is True:
            work_table['status'] = 'none'
            work_table['color'] = self.rainbow_colors[work_table.total_index % len(self.rainbow_colors)]
        else:
            # mark new, cont, end
            marker_color_new = self.marker_colors[0]
            marker_color_cont = self.marker_colors[1]
            marker_color_end = self.marker_colors[2]

            work_table['status'] = self.tracking_status(work_table)

            # make color list
            work_table['color'] = marker_color_cont
            work_table.loc[work_table['status'] == 'new', 'color'] = marker_color_new
            work_table.loc[work_table['status'] == 'end', 'color'] = marker_color_end
            work_table.loc[work_table['status'] == 'one', 'color'] = marker_color_new

            # modify spot colors in the first plane
            work_table.loc[(work_table['plane'] == 0) & (work_table['status'] == 'new'), 'color'] = marker_color_cont
            work_table.loc[(work_table['plane'] == 0) & (work_table['status'] == 'one'), 'color'] = marker_color_end
            #print(work_table)

        # use _regression
        if self.mark_regression is True:
            index_set = set(work_table[work_table.plane == 0].total_index)
            if self.force_mark_emerge is True:
                work_table = work_table[(work_table.total_index.isin(index_set)) | \
                                        ((work_table.life_index == 0) & (work_table.plane > 0))].reset_index(drop=True)                
                work_table.loc[(work_table.life_index == 0) & (work_table.plane > 0), 'status'] = 'emerge'
                work_table.loc[(work_table.life_index == 0) & (work_table.plane > 0), 'color'] = marker_color_new
            else:
                work_table = work_table[work_table.total_index.isin(index_set)].reset_index(drop=True)                

        # draw markers
        skipped_planes = []
        for index in range(len(image_color)):
            spots = work_table[work_table.plane == index].reset_index(drop=True)

            if len(spots) == 0:
                skipped_planes.append(index)
                continue

            spots['int_x'] = spots['x'].astype(numpy.int)
            spots['int_y'] = spots['y'].astype(numpy.int)
            spots = spots.sort_values(by = ['int_x', 'int_y']).reset_index(drop=True)

            # check possible error spots (duplicated)
            spots['duplicated'] = spots.duplicated(subset = ['int_x', 'int_y'], keep = False)
            error_spots = len(spots[spots['duplicated'] == True])
            if error_spots > 0:
                print("Possible %d duplicated spots in plane %d." % (error_spots, index))

            image = Image.fromarray(image_color[index])
            draw = ImageDraw.Draw(image)

            for row, spot in spots.iterrows():
                # draw marker
                if spot['status'] == 'emerge':
                    draw.rectangle(((spot.int_x - self.marker_size, spot.int_y - self.marker_size),\
                                   (spot.int_x + self.marker_size, spot.int_y + self.marker_size)),\
                                   fill = None, width = self.marker_width, outline = spot.color)
                else:
                    draw.ellipse(((spot.int_x - self.marker_size, spot.int_y - self.marker_size),\
                                (spot.int_x + self.marker_size, spot.int_y + self.marker_size)),\
                                fill = None, width = self.marker_width, outline = spot.color)

                    # draw additional marker
                    if spot['status'] == 'one':
                        draw.arc(((spot.int_x - self.marker_size, spot.int_y - self.marker_size),\
                                (spot.int_x + self.marker_size, spot.int_y + self.marker_size)),\
                                315, 135, width = self.marker_width, fill = marker_color_end)

                # mark duplicated spot
                if spot['duplicated'] is True:
                    draw.ellipse(((spot.int_x - self.marker_size + 1, spot.int_y - self.marker_size + 1),\
                                  (spot.int_x + self.marker_size + 1, spot.int_y + self.marker_size + 1)),\
                                  fill = None, width = self.marker_width, outline = self.marker_colors[3])

            image_color[index] = numpy.asarray(image)

        if sum(skipped_planes) > 0:
            print("Skipped planes %s." % (' '.join(map(str, skipped_planes))))

        return image_color
