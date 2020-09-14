#!/usr/bin/env python

import platform, sys, pathlib, numpy, pandas, time, decimal
from PIL import Image, ImageDraw, ImageFont

class FindArrow:
    def __init__ (self, arrow_filename, spot_shift = [0.0, 0.0], spot_scaling = 1.0):
        self.dist_threshold = None
        self.spot_shift = spot_shift
        self.spot_scaling = spot_scaling
        self.arrow_filename = arrow_filename
        self.arrow_table = self.read_arrow_table (arrow_filename)
        self.font_size = 20

    def load_font (self):
        if platform.system() == "Windows":
            font_filename = 'C:/Windows/Fonts/Arial.ttf'
        elif platform.system() == "Linux":
            font_filename = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
        elif platform.system() == "Darwin":
            font_filename = '/Library/Fonts/Verdana.ttf'
        else:
            raise Exception('Font file error.')

        return ImageFont.truetype(font_filename, self.font_size)

    def output_header (self, output_file):
        filename = pathlib.Path(self.arrow_filename).name
        output_file.write('## Output by FindArrow at {0} for {1}\n'.format(time.ctime(), filename))
        output_file.write('#   spot_shift = [{0:f}, {1:f}]; spot_scaling = {2:f}\n'.\
                          format(self.spot_shift[0], self.spot_shift[1], self.spot_scaling))

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

    def scale_spots (self, spot_table):
        work_table = spot_table.copy()
        work_table['x'] = (work_table['x'] * self.spot_scaling) + self.spot_shift[0]
        work_table['y'] = (work_table['y'] * self.spot_scaling) + self.spot_shift[1]
        return work_table

    def find_nearest_arrow (self, spot_table):
        # adjust coordinates
        work_table = self.scale_spots(spot_table)

        # find nearest arrow
        nearest_arrows = []
        key_list = ['total_index', 'arrow_index', 'arrow_x0', 'arrow_y0', 'arrow_x1', 'arrow_y1', \
                    'arrow_origin', 'arrow_dist0', 'arrow_dist1', 'arrow_dist', \
                    'arrow_pos0', 'arrow_pos1', 'arrow_pos']
        column_list = ['total_index', 'arrow_index', 'x0', 'y0', 'x1', 'y1', \
                        'origin', 'dist0', 'dist1', 'dist', \
                        'pos0', 'pos1', 'pos']

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

            #arrow_temp['dist0'] = numpy.amax((numpy.amin((arrow_temp['edge_dist0'], arrow_temp['edge_dist1']), axis = 0), arrow_temp['line_dist0']), axis = 0)
            #arrow_temp['dist1'] = numpy.amax((numpy.amin((arrow_temp['edge_dist0'], arrow_temp['edge_dist1']), axis = 0), arrow_temp['line_dist1']), axis = 0)
            arrow_temp['dist0'] = arrow_temp['line_dist0']
            arrow_temp['dist1'] = arrow_temp['line_dist1']
            arrow_temp['pos0'] = numpy.diag(numpy.dot(numpy.array((spot.x - arrow_temp.x0, spot.y - arrow_temp.y0)).T, \
                                                    numpy.array((arrow_temp.x1 - arrow_temp.x0, arrow_temp.y1 - arrow_temp.y0)))) \
                                / arrow_temp.length
            arrow_temp['pos1'] = arrow_temp.length - \
                                numpy.diag(numpy.dot(numpy.array((spot.x - arrow_temp.x1, spot.y - arrow_temp.y1)).T, \
                                            numpy.array((arrow_temp.x0 - arrow_temp.x1, arrow_temp.y0 - arrow_temp.y1)))) \
                                / arrow_temp.length

            arrow_temp['dist'] = numpy.nan
            arrow_temp.loc[arrow_temp.origin == 0, 'dist'] = arrow_temp['dist0']
            arrow_temp.loc[arrow_temp.origin == 1, 'dist'] = arrow_temp['dist1']
            arrow_temp['pos'] = numpy.nan
            arrow_temp.loc[arrow_temp.origin == 0, 'pos'] = arrow_temp['pos0']
            arrow_temp.loc[arrow_temp.origin == 1, 'pos'] = arrow_temp['pos1']

            arrow_temp = arrow_temp.sort_values('dist')

            nearest = {key: arrow_temp[column].to_list()[0] for (key, column) in zip(key_list, column_list)}
            nearest_arrows.append(nearest)

        nearest_arrows = {key: [arrow[key] for arrow in nearest_arrows] for key in key_list}
        
        return pandas.DataFrame(nearest_arrows, columns = key_list)

    def draw_for_arrow (self, back_image, spot_table):
        # adjust coordinates
        work_table = self.scale_spots(spot_table)

        output_images = []
        for index, arrow in self.arrow_table.iterrows():
            # arrow (R)
            arrow_image = Image.fromarray(numpy.zeros_like(back_image))
            draw = ImageDraw.Draw(arrow_image)
            arrow_xy = numpy.array([arrow.x0, arrow.y0, arrow.x1, arrow.y1], dtype = numpy.int)
            draw.line(tuple(arrow_xy), fill = 'white', width = 2)

            # spot (G)
            spot_image = Image.fromarray(numpy.zeros_like(back_image))
            draw = ImageDraw.Draw(spot_image)
            nearest_spot_table = work_table[work_table.arrow_index == index]
            for spot_index, spot in nearest_spot_table.iterrows():
                point_x = decimal.Decimal(spot.x).quantize(decimal.Decimal('0'), rounding=decimal.ROUND_HALF_UP)
                point_y = decimal.Decimal(spot.y).quantize(decimal.Decimal('0'), rounding=decimal.ROUND_HALF_UP)    
                draw.point((point_x, point_y), fill = 'white')

            # background (B): back_image.copy()

            output_images.append([numpy.asarray(arrow_image), numpy.asarray(spot_image), back_image.copy()])

        return numpy.array(output_images)

    def draw_for_spot (self, back_image, spot_table):
        # adjust coordinates
        work_table = self.scale_spots(spot_table)
        
        text_font = self.load_font()
        output_images = []
        for index, spot in work_table.iterrows():
            # arrow (R)
            arrow_image = Image.fromarray(numpy.zeros_like(back_image))
            draw = ImageDraw.Draw(arrow_image)
            for arrow_index, arrow in self.arrow_table.iterrows():
                arrow_xy = numpy.array([arrow.x0, arrow.y0, arrow.x1, arrow.y1], dtype = numpy.int)
                draw.line(tuple(arrow_xy), fill = 'white', width = 1)

            # spot (G)
            spot_image = Image.fromarray(numpy.zeros_like(back_image))
            draw = ImageDraw.Draw(spot_image)
            nearest_spot_table = spot_table[spot_table.arrow_index == index]
            point_x = decimal.Decimal(spot.x).quantize(decimal.Decimal('0'), rounding=decimal.ROUND_HALF_UP)
            point_y = decimal.Decimal(spot.y).quantize(decimal.Decimal('0'), rounding=decimal.ROUND_HALF_UP)    
            draw.point((point_x, point_y), fill = 'white')
            draw.text((0, 0), \
                      "Spot {0:d}, arrow {1:d}, dist {2:.2f} pos {3:.2f}".format(index, int(spot.arrow_index), spot.arrow_dist, spot.arrow_pos), \
                      font = text_font, fill = 'white')

            # arrow (B)
            one_arrow_image = Image.fromarray(numpy.zeros_like(back_image))
            draw = ImageDraw.Draw(one_arrow_image)
            arrow_xy = numpy.array([spot.arrow_x0, spot.arrow_y0, spot.arrow_x1, spot.arrow_y1], dtype = numpy.int)
            draw.line(tuple(arrow_xy), fill = 'white', width = 1)

            # background (C?): back_image.copy()

            output_images.append([numpy.asarray(arrow_image), numpy.asarray(spot_image), \
                                  numpy.asarray(one_arrow_image), back_image.copy()])
            
        return numpy.array(output_images)
