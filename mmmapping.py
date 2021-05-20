#!/usr/bin/env python

import sys, pathlib, re, argparse, json, numpy, pandas, czifile
import xml.etree.ElementTree as ET
from matplotlib import pyplot

# default values
input_locations = None
output_prefix = 'Mapping'
suffix_tsv_file = ".txt"
suffix_graph_file = ".png"
axis_mode = 'auto'

# check if the data locations can be treated
def acceptable (location):
    path = pathlib.Path(location)
    if not path.exists():
        return False
    elif path.is_dir():
        if path.glob("*_metadata.txt"):
            return True
        else:
            return False
    elif path.suffix.lower() == ".czi":
        return True

    return False

# parse arguments
parser = argparse.ArgumentParser(description='Map the localization of cells', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-prefix', default=output_prefix, \
                    help='prefix of output files')

parser.add_argument('-a', '--axis-mode', default=axis_mode, choices = ['auto', 'dispim', 'airy'], \
                    help='set the directions of axis')

parser.add_argument('input_locations', nargs = '*', default=input_locations, \
                    help='folders or czi files to process (fetch all for None)')
args = parser.parse_args()

# set arguments
output_prefix = args.output_prefix
input_locations = args.input_locations
axis_mode = args.axis_mode

output_tsv_filename = output_prefix + suffix_tsv_file
output_graph_filename = output_prefix + suffix_graph_file

if input_locations is None or len(input_locations) == 0:
    input_locations = list(filter(lambda x: acceptable(x), pathlib.Path(".").glob("*")))
else:
    locations = []
    for input_location in input_locations:
        locations.extend(list(pathlib.Path(".").glob(input_location)))
    input_locations = list(filter(lambda x: acceptable(x), locations))

if len(input_locations) == 0:
    print("No input folders. Exit.")
    sys.exit()

positions = []
for index, input_location in enumerate(input_locations):
    if input_location.is_dir():
        metafile = next(pathlib.Path(input_location).glob("*_metadata.txt"))
        with metafile.open('r', encoding = 'iso-8859-1') as file:
            data = json.load(file)
            pattern = re.compile(r'[-+]?[0-9,]*\.?[0-9,]+([eE][-+]?[0-9]+)?')
            pos_x = float(pattern.match(data["Summary"]["Position_X"]).group().replace(',', ''))
            pos_y = float(pattern.match(data["Summary"]["Position_Y"]).group().replace(',', ''))
    elif input_location.suffix.lower() == ".czi":
        with czifile.CziFile(str(input_location)) as file:
            root = ET.fromstring(file.metadata())
            element = root.find('./Metadata/Information/Image/Dimensions/S/Scenes/Scene[@Index="0"]/Positions/Position')
            pos_x = float(element.attrib['X'])
            pos_y = float(element.attrib['Y'])
    else:
        print("Ignoring:", str(input_location))
        continue
    
    positions.append({'index': index, 'name': input_location, 'x': pos_x, 'y': pos_y})

pos_table = pandas.DataFrame(positions)
if len(pos_table) == 0:
    print("Empty table. No output.")
    sys.exit()

print("Output TSV file:", output_tsv_filename)
print(pos_table)
pos_table.to_csv(output_tsv_filename, sep = "\t")

print("Output graph file:", output_graph_filename)
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Locations of samples", size = 'xx-large')
axes.set_xlabel("X (um)", size = 'xx-large')
axes.set_ylabel("Y (um)", size = 'xx-large')

# set the direction of axis
x_min = (numpy.min(pos_table.x) // 1000) * 1000
x_max = (numpy.max(pos_table.x) // 1000 + 1) * 1000
y_min = (numpy.min(pos_table.y) // 1000) * 1000
y_max = (numpy.max(pos_table.y) // 1000 + 1) * 1000

dirs = list(filter(lambda x: x.is_dir(), input_locations))
czis = list(filter(lambda x: x.suffix.lower() == '.czi', input_locations))

if axis_mode == 'auto':
    if len(dirs) >= len(czis):
        axis_mode = 'dispim'
        print("Axis direction set to the diSPIM mode")
    else:
        axis_mode = 'airy'
        print("Axis direction set to the Airyscan mode")

if axis_mode == 'dispim':
    if len(czis) > 0:
        print("Warning: some CZI files are processed.")
    axes.set_xlim(x_max, x_min)
    axes.set_ylim(y_max, y_min)
else:
    if len(dirs) > 0:
        print("Warning: some diSPIM files are processed.")
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_max, y_min)

axes.axhline(0, color = 'black', linestyle = ':')
axes.axvline(0, color = 'black', linestyle = ':')

for index, item in pos_table.iterrows():
    axes.plot(item['x'], item['y'], marker = 'o', label = item['name'])
    #axes.annotate(index, (item['x'], item['y']))

axes.legend()
figure.savefig(output_graph_filename, dpi = 300)
