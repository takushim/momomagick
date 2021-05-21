#!/usr/bin/env python

import platform, sys, pathlib, numpy, pandas, time
from . import mmtiff
from PIL import Image, ImageDraw, ImageFont

class FindArrow:
    def __init__ (self, arrow_filename):
        self.dist_threshold = None
        self.arrow_filename = arrow_filename
        self.arrow_table = self.read_arrow_table(arrow_filename)
        self.font_size = 20
        self.max_distance = None

    def read_arrow_table (self, arrow_filename):
        # read an ImageJ arrow table (pixel unit)
        arrow_table = pandas.read_csv(arrow_filename, comment = '#')

        # add coordinates of arrows
        arrow_table['arrow_index'] = numpy.arange(len(arrow_table))
        arrow_table['x0'] = numpy.nan
        arrow_table['y0'] = numpy.nan
        arrow_table['x1'] = numpy.nan
        arrow_table['y1'] = numpy.nan

        arrow_table.loc[(arrow_table.Angle >= 0) & (arrow_table.Angle < 90), 'x0'] = arrow_table['BX']
        arrow_table.loc[(arrow_table.Angle >= 0) & (arrow_table.Angle < 90), 'y0'] = arrow_table['BY'] + arrow_table['Height']
        arrow_table.loc[(arrow_table.Angle >= 0) & (arrow_table.Angle < 90), 'x1'] = arrow_table['BX'] + arrow_table['Width']
        arrow_table.loc[(arrow_table.Angle >= 0) & (arrow_table.Angle < 90), 'y1'] = arrow_table['BY']

        arrow_table.loc[(arrow_table.Angle >= 90) & (arrow_table.Angle <= 180), 'x0'] = arrow_table['BX'] + arrow_table['Width']
        arrow_table.loc[(arrow_table.Angle >= 90) & (arrow_table.Angle <= 180), 'y0'] = arrow_table['BY'] + arrow_table['Height']
        arrow_table.loc[(arrow_table.Angle >= 90) & (arrow_table.Angle <= 180), 'x1'] = arrow_table['BX']
        arrow_table.loc[(arrow_table.Angle >= 90) & (arrow_table.Angle <= 180), 'y1'] = arrow_table['BY']

        arrow_table.loc[(arrow_table.Angle < 0) & (arrow_table.Angle >= -90), 'x0'] = arrow_table['BX']
        arrow_table.loc[(arrow_table.Angle < 0) & (arrow_table.Angle >= -90), 'y0'] = arrow_table['BY']
        arrow_table.loc[(arrow_table.Angle < 0) & (arrow_table.Angle >= -90), 'x1'] = arrow_table['BX'] + arrow_table['Width']
        arrow_table.loc[(arrow_table.Angle < 0) & (arrow_table.Angle >= -90), 'y1'] = arrow_table['BY'] + arrow_table['Height']

        arrow_table.loc[(arrow_table.Angle < -90) & (arrow_table.Angle >= -180), 'x0'] = arrow_table['BX'] + arrow_table['Width']
        arrow_table.loc[(arrow_table.Angle < -90) & (arrow_table.Angle >= -180), 'y0'] = arrow_table['BY']
        arrow_table.loc[(arrow_table.Angle < -90) & (arrow_table.Angle >= -180), 'x1'] = arrow_table['BX']
        arrow_table.loc[(arrow_table.Angle < -90) & (arrow_table.Angle >= -180), 'y1'] = arrow_table['BY'] + arrow_table['Height']

        return arrow_table

    def find_nearest_arrow (self, spot_table):
        work_table = spot_table.copy()

        # find nearest arrow
        nearest_arrows = []
        key_list = ['total_index', 'arrow_index', 'arrow_length', 'arrow_x0', 'arrow_y0', 'arrow_x1', 'arrow_y1', \
                    'arrow_origin', 'arrow_dist_e0', 'arrow_dist_e1', 'arrow_dist_l0', 'arrow_dist_l1', \
                    'arrow_dist_line', 'arrow_dist_min', 'arrow_pos0', 'arrow_pos1', 'arrow_pos']
        column_list = ['total_index', 'arrow_index', 'length', 'x0', 'y0', 'x1', 'y1', \
                       'origin', 'edge_dist0', 'edge_dist1', 'line_dist0', 'line_dist1', \
                       'dist_line', 'dist_min', 'pos0', 'pos1', 'pos']

        for index, spot in work_table.iterrows():
            arrow_temp = self.arrow_table.copy()
            arrow_temp['total_index'] = index
            arrow_temp['length'] = numpy.linalg.norm((arrow_temp.x0 - arrow_temp.x1, arrow_temp.y0 - arrow_temp.y1), axis = 0)
            arrow_temp['edge_dist0'] = numpy.linalg.norm((spot.x - arrow_temp.x0, spot.y - arrow_temp.y0), axis = 0)
            arrow_temp['edge_dist1'] = numpy.linalg.norm((spot.x - arrow_temp.x1, spot.y - arrow_temp.y1), axis = 0)
            arrow_temp['line_dist0'] = numpy.abs(numpy.cross((spot.x - arrow_temp.x0, spot.y - arrow_temp.y0), \
                                                            (arrow_temp.x1 - arrow_temp.x0, arrow_temp.y1 - arrow_temp.y0), axis = 0)) \
                                    / numpy.linalg.norm((arrow_temp.x1 - arrow_temp.x0, arrow_temp.y1 - arrow_temp.y0), axis = 0)
            arrow_temp['line_dist1'] = numpy.abs(numpy.cross((spot.x - arrow_temp.x1, spot.y - arrow_temp.y1), \
                                                            (arrow_temp.x1 - arrow_temp.x0, arrow_temp.y1 - arrow_temp.y0), axis = 0)) \
                                    / numpy.linalg.norm((arrow_temp.x1 - arrow_temp.x0, arrow_temp.y1 - arrow_temp.y0), axis = 0)

            arrow_temp['origin'] = 0
            arrow_temp.loc[arrow_temp.edge_dist0 < arrow_temp.edge_dist1, 'origin'] = 1

            arrow_temp['pos0'] = numpy.diag(numpy.dot(numpy.array((spot.x - arrow_temp.x0, spot.y - arrow_temp.y0)).T, \
                                                      numpy.array((arrow_temp.x1 - arrow_temp.x0, arrow_temp.y1 - arrow_temp.y0)))) \
                                / arrow_temp.length
            arrow_temp['pos1'] = arrow_temp.length - \
                                numpy.diag(numpy.dot(numpy.array((spot.x - arrow_temp.x1, spot.y - arrow_temp.y1)).T, \
                                                     numpy.array((arrow_temp.x0 - arrow_temp.x1, arrow_temp.y0 - arrow_temp.y1)))) \
                                / arrow_temp.length

            arrow_temp['dist_line'] = numpy.nan
            arrow_temp.loc[arrow_temp.origin == 0, 'dist_line'] = arrow_temp['line_dist0']
            arrow_temp.loc[arrow_temp.origin == 1, 'dist_line'] = arrow_temp['line_dist1']
            arrow_temp['pos'] = numpy.nan
            arrow_temp.loc[arrow_temp.origin == 0, 'pos'] = arrow_temp['pos0']
            arrow_temp.loc[arrow_temp.origin == 1, 'pos'] = arrow_temp['pos1']

            arrow_temp['dist_min'] = arrow_temp['dist_line']
            arrow_temp.loc[arrow_temp.pos < 0, 'dist_min'] = arrow_temp['edge_dist0']
            arrow_temp.loc[arrow_temp.pos > arrow_temp.length, 'dist_min'] = arrow_temp['edge_dist1']

            arrow_temp = arrow_temp.sort_values('dist_line')

            nearest = {key: arrow_temp[column].to_list()[0] for (key, column) in zip(key_list, column_list)}
            nearest_arrows.append(nearest)

        nearest_arrows = {key: [arrow[key] for arrow in nearest_arrows] for key in key_list}
        nearest_arrow_table = pandas.DataFrame(nearest_arrows, columns = key_list)
        if self.max_distance is not None:
            nearest_arrow_table.loc[nearest_arrow_table.arrow_dist_min > self.max_distance, 'arrow_index'] = -1
        
        return nearest_arrow_table

    def draw_arrows (self, back_image, arrow_table, arrow_index = None):
        image = Image.fromarray(numpy.zeros_like(back_image, dtype = numpy.int8))
        draw = ImageDraw.Draw(image)

        if arrow_index is None:
            arrows_to_draw = arrow_table
        else:
            arrows_to_draw = arrow_table[arrow_table.arrow_index == arrow_index]

        for arrow in arrows_to_draw.itertuples():
            arrow_xy = numpy.array([arrow.x0, arrow.y0, arrow.x1, arrow.y1])
            draw.line(tuple(numpy.round(arrow_xy)), fill = 'white', width = 1)
            #draw.ellipse((arrow.x0 - 2, arrow.y0 - 2, arrow.x0 + 2, arrow.y0 + 2), fill = None, outline = 'white')

        return numpy.array(image)

    def draw_spots (self, back_image, spot_table, spot_index = None, arrow_index = None):
        text_font = ImageFont.truetype(mmtiff.font_path(), self.font_size)
        image = Image.fromarray(numpy.zeros_like(back_image)) # int8 fails... why?
        draw = ImageDraw.Draw(image)

        if spot_index is None:
            if arrow_index is None:
                spots_to_draw = spot_table
                draw_text = "Total {0} spots".format(len(spots_to_draw))
            elif arrow_index >= 0:
                spots_to_draw = spot_table[spot_table.arrow_index == arrow_index]
                draw_text = "Arrow {0}: total {1} spots".format(arrow_index, len(spots_to_draw))
            else:
                spots_to_draw = spot_table[spot_table.arrow_index < 0]
                draw_text = "Arrow unassigned: total {0} spots".format(len(spots_to_draw))
        else:
            if arrow_index is None:
                spots_to_draw = spot_table[spot_table.index == spot_index]
                lifetime = spots_to_draw.life_total.to_list()[0]
                draw_text = "Spot {0}: dwells {1} frames".format(spot_index, lifetime)
            else:
                raise Exception("Either spot_index or arrow_index should be specified.")

        for spot in spots_to_draw.itertuples():
            point_xy = numpy.array([spot.x, spot.y])
            draw.point(tuple(numpy.round(point_xy)), fill = 'white')
        draw.text((0, 0), draw_text, font = text_font, color = 'white')

        return numpy.array(image)

    def draw_for_arrow (self, back_image, spot_table):
        work_table = spot_table.copy()
        output_images = []

        # plot of all arrows
        arrow_image = self.draw_arrows(back_image, self.arrow_table)
        spot_image = self.draw_spots(back_image, work_table)
        output_images.append([arrow_image, spot_image, back_image.copy()])

        # plot for each arrow
        for index in range(len(self.arrow_table)):
            arrow_image = self.draw_arrows(back_image, self.arrow_table, index)
            spot_image = self.draw_spots(back_image, work_table, None, index)
            output_images.append([arrow_image, spot_image, back_image.copy()])

        # plot of unassigned spots
        if len(work_table[work_table.arrow_index < 0]) > 0:
            arrow_image = self.draw_arrows(back_image, self.arrow_table)
            spot_image = self.draw_spots(back_image, work_table, None, -1)
            output_images.append([arrow_image, spot_image, back_image.copy()])

        return numpy.array(output_images, dtype = back_image.dtype)

    def draw_for_spot (self, back_image, spot_table):
        work_table = spot_table.copy()
        
        output_images = []
        for index, spot in enumerate(work_table.itertuples()):
            arrow_image = self.draw_arrows(back_image, self.arrow_table)
            spot_image = self.draw_spots(back_image, work_table, index)
            if spot.arrow_index >= 0:
                one_arrow_image = self.draw_arrows(back_image, self.arrow_table, spot.arrow_index)
            else:
                one_arrow_image = numpy.zeros_like(back_image)
            output_images.append([arrow_image, spot_image, one_arrow_image, back_image.copy()])
            
        return numpy.array(output_images, dtype = back_image.dtype)
