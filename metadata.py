#!/usr/bin/env python

import os, sys, glob, argparse, json, pprint

# default values
input_filename = None
frame_number = None

# parse arguments
parser = argparse.ArgumentParser(description='Show formatted metadata of diSPIM image', \
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-n', '--frame-number', nargs=1, type=int, default=frame_number, \
                    help='show metadata of specified frame (begins with 0)')
parser.add_argument('input_file', nargs='?', default=input_filename, \
                    help='name of metadata file')
args = parser.parse_args()

# set arguments
if args.input_file is None:
    input_filename = sorted(glob.glob('*metadata.txt'))[0]
    print("Reading: %s" % (input_filename))
else:
    input_filename = args.input_file

if args.frame_number is not None:
    frame_number = args.frame_number[0]

# read json
with open(input_filename, 'r') as file:
    json_data = json.load(file)

# pprint
pprint.pprint(json_data['Summary'])

if frame_number is not None:
    key = "FrameKey-%d-0-0" % (frame_number)
    pprint.pprint(json_data[key])