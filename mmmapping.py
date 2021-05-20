#!/usr/bin/env python

import sys, pathlib, re, argparse, json, numpy, pandas
from matplotlib import pyplot

# default values
input_folders = None
output_prefix = 'Mapping'
suffix_tsv_file = ".txt"
suffix_graph_file = ".png"

# parse arguments
parser = argparse.ArgumentParser(description='Map the localization of cells', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o', '--output-prefix', default=output_prefix, \
                    help='prefix of output files')

parser.add_argument('input_folders', nargs = '*', default=input_folders, \
                    help='folders to process (fetch all for None)')
args = parser.parse_args()

# set arguments
output_prefix = args.output_prefix
input_folders = args.input_folders

output_tsv_filename = output_prefix + suffix_tsv_file
output_graph_filename = output_prefix + suffix_graph_file


if input_folders is None or len(input_folders) == 0:
    input_folders = filter(lambda x: pathlib.Path(x).is_dir(), pathlib.Path(".").glob("*"))
    input_folders = [str(x) for x in input_folders]
else:
    temp_folders = []
    for input_folder in input_folders:
        temp_folders.extend(list(pathlib.Path(".").glob(input_folder)))
    input_folders = [str(x) for x in filter(lambda x: x.is_dir() and x.exists(), temp_folders)]

if len(input_folders) == 0:
    print("No input folders. Exit.")
    sys.exit()

positions = []
for index, input_folder in enumerate(input_folders):
    metafile = [str(x) for x in pathlib.Path(input_folder).glob("*_metadata.txt")][0]
    with open(metafile, 'r') as file:
        data = json.load(file)
        pattern = re.compile(r'[-+]?[0-9,]*\.?[0-9,]+([eE][-+]?[0-9]+)?')
        pos_x = float(pattern.match(data["Summary"]["Position_X"]).group().replace(',', ''))
        pos_y = float(pattern.match(data["Summary"]["Position_Y"]).group().replace(',', ''))
    positions.append({'index': index, 'name': input_folder, 'x': pos_x, 'y': pos_y})

pos_table = pandas.DataFrame(positions)

print("Output TSV file:", output_tsv_filename)
print(pos_table)
pos_table.to_csv(output_tsv_filename, sep = "\t")

print("Output graph file:", output_graph_filename)
figure = pyplot.figure(figsize = (12, 8), dpi = 300)
axes = figure.add_subplot(111)
axes.set_title("Locations of samples", size = 'xx-large')
axes.set_xlabel("X (um)", size = 'xx-large')
axes.set_ylabel("Y (um)", size = 'xx-large')

x_bound = (numpy.abs(pos_table.x).max() // 1000 + 1) * 1000
y_bound = (numpy.abs(pos_table.y).max() // 1000 + 1) * 1000
axes.set_xlim(x_bound, -x_bound)
axes.set_ylim(y_bound, -y_bound)
axes.axhline(0, color = 'black', linestyle = ':')
axes.axvline(0, color = 'black', linestyle = ':')

axes.scatter(pos_table.x, pos_table.y, c = pos_table.index)
for index, item in pos_table.iterrows():
    axes.annotate(index, (item['x'], item['y']))
figure.savefig(output_graph_filename, dpi = 300)
