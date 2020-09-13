#!/usr/bin/env python

import os, platform, sys, argparse, pathlib, re, numpy, pandas, tifffile, pprint, decimal
from mmtools import mmtiff
from PIL import Image, ImageDraw, ImageFont

scaling = 2.0
arrow_table = pandas.read_csv('Results.csv', comment = '#')
spot_table = pandas.read_csv('Cell1_500msLp0.4_fix_nobleach_0_aligned_2x_reconv.txt', comment = '#', sep = '\t')
spot_table['x'] = spot_table['x'] / scaling
spot_table['y'] = spot_table['y'] / scaling

spot_table['x'] = spot_table['x'] -7.5
spot_table['y'] = spot_table['y'] + 22.5
spot_table = spot_table.sort_values(['total_index', 'plane']).drop_duplicates(subset='total_index', keep='first').reset_index(drop=True)

orig_tiff = mmtiff.MMTiff("Cell1_EGFP_1_max.tif")
orig_image = orig_tiff.as_array()[0, 0, 0]

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

#print(arrow_table)

# find nearest arrow
dist_threshold = 10000
nearest_arrows = []
key_list = ['total_index', 'arrow_index', 'arrow_x0', 'arrow_y0', 'arrow_x1', 'arrow_y1', \
            'arrow_origin', 'arrow_dist0', 'arrow_dist1', 'arrow_dist', \
            'arrow_pos0', 'arrow_pos1', 'arrow_pos']
column_list = ['total_index', 'arrow_index', 'x0', 'y0', 'x1', 'y1', \
                'origin', 'dist0', 'dist1', 'dist', \
                'pos0', 'pos1', 'pos']

for index, spot in spot_table.iterrows():
    arrow_temp = arrow_table.copy()
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
    arrow_temp.to_csv("temp/spot_{0:04d}.txt".format(index), sep = "\t", index = False)

    nearest = {key: arrow_temp[column].to_list()[0] for (key, column) in zip(key_list, column_list)}
    nearest_arrows.append(nearest)

nearest_arrows = {key: [arrow[key] for arrow in nearest_arrows] for key in key_list}
nearest_arrow_table = pandas.DataFrame(nearest_arrows, columns = key_list)

spot_table = pandas.merge(spot_table, nearest_arrow_table, left_on='total_index', right_on='total_index', how='left')
spot_table['arrow_pos_um'] = spot_table['arrow_pos'] * 0.1625
spot_table.to_csv("Spot_output.txt", sep = "\t", index = False)

# draw lines
output_images = []
marker_size = 1
for index, arrow in arrow_table.iterrows():
    # arrow (R)
    arrow_image = Image.fromarray(numpy.zeros_like(orig_image))
    draw = ImageDraw.Draw(arrow_image)
    arrow_xy = numpy.array([arrow.x0, arrow.y0, arrow.x1, arrow.y1], dtype = numpy.int)
    draw.line(tuple(arrow_xy), fill = 'white', width = 2)
    #draw.ellipse((arrow_xy[0] - marker_size, arrow_xy[1] - marker_size, \
    #              arrow_xy[0] + marker_size, arrow_xy[1] + marker_size), \
    #             width = 1, fill = None, outline = 'white')

    # spot (G)
    spot_image = Image.fromarray(numpy.zeros_like(orig_image))
    draw = ImageDraw.Draw(spot_image)
    nearest_spot_table = spot_table[spot_table.arrow_index == index]
    for spot_index, spot in nearest_spot_table.iterrows():
        draw.ellipse((spot.x - marker_size, spot.y - marker_size, \
                      spot.x + marker_size, spot.y + marker_size), \
                     width = 1, fill = None, outline = 'white')

    output_images.append([numpy.asarray(arrow_image), numpy.asarray(spot_image), orig_image.copy()])

# output ImageJ, dimensions should be in TZCYXS order
tifffile.imsave("EGFP_G63D_arrows.tif", numpy.array(output_images), imagej = True, \
                resolution = (1 / orig_tiff.pixelsize_um, 1 / orig_tiff.pixelsize_um), \
                metadata = {'spacing': orig_tiff.z_step_um, 'unit': 'um', 'Composite mode': 'composite'})

# for debug
if platform.system() == "Windows":
    font_file = 'C:/Windows/Fonts/Arial.ttf'
elif platform.system() == "Linux":
    font_file = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
elif platform.system() == "Darwin":
    font_file = '/Library/Fonts/Verdana.ttf'
else:
    raise Exception('font file error.')
font = ImageFont.truetype(font_file, 40)

output_images = []
for index, spot in spot_table.iterrows():
    # arrow (R)
    arrow_image = Image.fromarray(numpy.zeros_like(orig_image))
    draw = ImageDraw.Draw(arrow_image)
    for arrow_index, arrow in arrow_table.iterrows():
        arrow_xy = numpy.array([arrow.x0, arrow.y0, arrow.x1, arrow.y1], dtype = numpy.int)
        draw.line(tuple(arrow_xy), fill = 'white', width = 1)

    # spot (G)
    spot_image = Image.fromarray(numpy.zeros_like(orig_image))
    draw = ImageDraw.Draw(spot_image)
    nearest_spot_table = spot_table[spot_table.arrow_index == index]
    point_x = decimal.Decimal(spot.x).quantize(decimal.Decimal('0'), rounding=decimal.ROUND_HALF_UP)
    point_y = decimal.Decimal(spot.y).quantize(decimal.Decimal('0'), rounding=decimal.ROUND_HALF_UP)    
    draw.point((point_x, point_y), fill = 'white')
    #draw.ellipse((spot.x - marker_size, spot.y - marker_size, \
    #                spot.x + marker_size, spot.y + marker_size), \
    #               width = 1, fill = None, outline = 'white')
    draw.text((0, 0), "Spot {0:d}, arrow {1:d}, dist {2:.2f} pos {3:.2f}".format(index, int(spot.arrow_index), spot.arrow_dist, spot.arrow_pos), font = font, fill = 'white')

    # arrow (R)
    arrow_image2 = Image.fromarray(numpy.zeros_like(orig_image))
    draw = ImageDraw.Draw(arrow_image2)
    arrow_xy = numpy.array([spot.arrow_x0, spot.arrow_y0, spot.arrow_x1, spot.arrow_y1], dtype = numpy.int)
    draw.line(tuple(arrow_xy), fill = 'white', width = 1)

    output_images.append([numpy.asarray(arrow_image), numpy.asarray(spot_image), numpy.asarray(arrow_image2)])

# output ImageJ, dimensions should be in TZCYXS order
tifffile.imsave("EGFP_G63D_arrows_debug.tif", numpy.array(output_images), imagej = True, \
                resolution = (1 / orig_tiff.pixelsize_um, 1 / orig_tiff.pixelsize_um), \
                metadata = {'spacing': orig_tiff.z_step_um, 'unit': 'um', 'Composite mode': 'composite'})
