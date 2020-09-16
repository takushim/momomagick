#!/usr/bin/env python

import sys, numpy, pandas, time

class TrackJ:
    def __init__ (self, filename):
        self.columns = ['total_index', 'plane', 'x', 'y']
        self.input_filename = filename
        self.spot_table = self.read_spots(filename)

    def output_header (self, output_file):
        output_file.write('## Reconverted by mmtrackj_reconvert at {0}\n'.format(time.ctime()))
        output_file.write('#   file = \'{0}\'; reference = %s\n'.format(self.input_filename))

    def read_spots (self, input_filename):
        # read lines
        input_file = open(input_filename, 'r')
        trackj_lines = input_file.readlines()
        trackj_lines = [line.rstrip() for line in trackj_lines]
        input_file.close()

        # parse TrackJ lines
        total_index = 0
        spot_list = []
        for line in trackj_lines:
            if line.startswith('#'):
                if line.startswith('#%stop'):
                    total_index = total_index + 1
            else:
                items = line.split()
                x, y, plane = float(items[0]), float(items[1]), int(items[2])
                spot_list.append([total_index, plane - 1, x, y])
        
        # convert to a dataframe
        spot_table = pandas.DataFrame(data = spot_list, columns = self.columns)
    
        return spot_table

    @staticmethod
    def save_spots (output_filename, spot_table):
        headers = ["#speckles csv ver 1.2\n", \
                   "#x(double)\ty(double)\tsize(double)\tframe(int)\ttype(int)\n"]

        # increment plane number
        work_table = spot_table.copy()
        work_table['plane'] = work_table['plane'] + 1

        # output
        output_file = open(output_filename, 'w', newline='')
        for header in headers:
            output_file.write(header)

        for index, spots in work_table.groupby('total_index'):
            output_file.write("#%start speckle%" + '\n')
            spots.to_csv(output_file, columns = ['x', 'y', 'plane'], sep='\t', index=False, header=False, mode='a')
            output_file.write("#%stop speckle%" + '\n')
        
        output_file.close()


