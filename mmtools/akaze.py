#!/usr/bin/env python

import sys, numpy, pandas, time, cv2
from . import spotshift

class Akaze:
    def __init__ (self):
        self.columns = ['align_plane', 'align_x', 'align_y']
        self.threshold = 0.00005
        self.matching_ratio = 0.15
        self.use_ransac = False

    def output_header (self, output_file, input_filename, reference_filename):
        output_file.write('## Alignment by TaniAlign (AKAZE) at %s\n' % (time.ctime()))
        output_file.write('#   file = \'%s\'; reference = %s\n' % (input_filename, reference_filename))
        output_file.write('#   threshold = %f; matching_ratio = %f\n' % (self.threshold, self.matching_ratio))

    def calculate_alignments (self, orig_images, reference = None):
        detector = cv2.AKAZE_create(threshold = self.threshold)

        # params of original image
        if reference is not None:
            (orig_kps, orig_descs) = detector.detectAndCompute(reference, None)
        else:
            (orig_kps, orig_descs) = detector.detectAndCompute(orig_images[0], None)

        results = []
        for index in range(len(orig_images)):
            # params of image
            (this_kps, this_descs) = detector.detectAndCompute(orig_images[index], None)

            # brute-force matching
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(orig_descs, this_descs, None)
            matches.sort(key = lambda x: x.distance, reverse = False)
            matches = matches[:(int(len(matches) * self.matching_ratio))]

            # calculate the movements of matching points
            orig_points = numpy.zeros((len(matches), 2), dtype = numpy.float32)
            this_points = numpy.zeros((len(matches), 2), dtype = numpy.float32)
            for i, match in enumerate(matches):
                orig_points[i, :] = orig_kps[match.queryIdx].pt
                this_points[i, :] = this_kps[match.trainIdx].pt

            # reduce error matching by RANSAC
            matching_method = 'RANSAC'
            h, masks = cv2.findHomography(orig_points, this_points, cv2.RANSAC, 3.0)
            masks = masks[:, 0]
            if numpy.all(masks[0] == 0):
                masks = numpy.ones_like(masks, dtype = numpy.int)
                matching_method = 'Brute-Force'

            # calculate the drift
            mvx = 0
            mvy = 0
            cnt = 0

            for point_index in numpy.where(masks == 1)[0]:
                mvx += this_points[point_index][0] - orig_points[point_index][0]
                mvy += this_points[point_index][1] - orig_points[point_index][1]
                cnt += 1

            mvx = mvx / cnt
            mvy = mvy / cnt
            print("Plane {0:d}, dislocation = ({1:f}, {2:f}) using {3}.".format(index, mvx, mvy, matching_method))

            results.append([index, mvx, mvy])

        # prepare a table
        align_plane = numpy.array([result[0] for result in results])
        align_x = numpy.array([result[1] for result in results])
        align_y = numpy.array([result[2] for result in results])
        align_table = pandas.DataFrame({self.columns[0]: align_plane, \
                                       self.columns[1]: align_x, \
                                       self.columns[2]: align_y, columns = self.columns

        # add smoothing and return
        return spotshift.SpotShift.add_smoothing(align_table)
