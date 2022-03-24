#!/usr/bin/env python

import platform
from PIL import ImageFont
from logging import getLogger

logger = getLogger(__name__)

def font_path ():
    if platform.system() == "Windows":
        font_filename = 'C:/Windows/Fonts/Arial.ttf'
    elif platform.system() == "Linux":
        font_filename = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
    elif platform.system() == "Darwin":
        font_filename = '/Library/Fonts/Verdana.ttf'
    else:
        raise Exception('Unknown operating system. Font cannot be loaded.')

    return font_filename

def mark_spots_func (radius, linewidth = 1, shape = 'circle'):
    if shape == 'circle':
        def mark_spots (draw, spots, color):
            for spot in spots:
                draw.ellipse((spot['x'] - radius, spot['y'] - radius, spot['x'] + radius, spot['y'] + radius),
                            outline = color, fill = None, width = linewidth)
    elif shape == 'dot':
        def mark_spots (draw, spots, color):
            for spot in spots:
                draw.point((spot['x'], spot['y']), outline = color, fill = None)
    elif shape == 'rectangle':
        def mark_spots (draw, spots, color):
            for spot in spots:
                draw.rectangle((spot['x'] - radius, spot['y'] - radius, spot['x'] + radius, spot['y'] + radius),
                            outline = color, fill = None, width = linewidth)
    elif shape == 'cross':
        def mark_spots (draw, spots, color):
            for spot in spots:
                draw.line((spot['x'] - radius, spot['y'] - radius, spot['x'] + radius, spot['y'] + radius),
                        fill = color, width = linewidth)
                draw.line((spot['x'] + radius, spot['y'] - radius, spot['x'] - radius, spot['y'] + radius),
                        fill = color, width = linewidth)
    elif shape == 'plus':
        def mark_spots (draw, spots, color):
            for spot in spots:
                draw.line((spot['x'] - radius, spot['y'], spot['x'] + radius, spot['y']),
                        fill = color, width = linewidth)
                draw.line((spot['x'], spot['y'] - radius, spot['x'], spot['y'] + radius),
                        fill = color, width = linewidth)
    elif shape == 'arc':
        def mark_spots (draw, spots, color):
            for spot in spots:
                draw.arc((spot['x'] - radius, spot['y'] - radius, spot['x'] + radius, spot['y'] + radius),
                        start = 315, end = 135, fill = color, width = linewidth)
    else:
        raise Exception('Unknown shape: {0}'.format(shape))

    return mark_spots

def draw_texts_func (offset, font_size, corner = 'NE'):
    font = ImageFont.truetype(font_path(), font_size)
    if corner == 'NE':
        def center_to_offset (spot, text_width, text_height):
            x = spot['x'] + offset
            y = spot['y'] - offset - text_height
            return (x, y)
    elif corner == 'NW':
        def center_to_offset (spot, text_width, text_height):
            x = spot['x'] - offset - text_width
            y = spot['y'] - offset - text_height
            return (x, y)
    elif corner == 'SE':
        def center_to_offset (spot, text_width, text_height):
            x = spot['x'] + offset
            y = spot['y'] + offset
            return (x, y)
    elif corner == 'SW':
        def center_to_offset (spot, text_width, text_height):
            x = spot['x'] - offset - text_width
            y = spot['y'] + offset
            return (x, y)
    else:
        raise Exception("Unknown corner specification: {0}".format(corner))

    def draw_texts (draw, spots, color):
        for spot in spots:
            text = spot.get("text", None)
            if text is None:
                continue
            width, height = font.getsize(text)
            draw.text(center_to_offset(spot, width, height), text, font = font, fill = color)

    return draw_texts
