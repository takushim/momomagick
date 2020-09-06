#!/usr/bin/env python

import sys, numpy, pandas, time, cv2
from statsmodels.nonparametric.smoothers_lowess import lowess

class Akaze:
    def __init__ (self):
        self.columns = ['align_plane', 'align_x', 'align_y', 'smooth_x', 'smooth_y']
        self.threshold = 0.00005
        self.matching_ratio = 0.15

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
            h, mask = cv2.findHomography(orig_points, this_points, cv2.RANSAC, 3.0)
            matches_mask = mask.ravel().tolist() # does this need?

            # calculate the drift
            mvx=0
            mvy=0
            cnt=0
            for k, v in enumerate(mask):
                if v==1:
                    mvx += this_points[k][0] - orig_points[k][0]
                    mvy += this_points[k][1] - orig_points[k][1]
                    cnt += 1
            mvx = mvx / cnt
            mvy = mvy / cnt
            print("Plane %d, dislocation = (%f, %f)." % (index, mvx, mvy))

            results.append([index, mvx, mvy])

        # add smoothing
        align_plane = numpy.array([result[0] for result in results])
        align_x = numpy.array([result[1] for result in results])
        align_y = numpy.array([result[2] for result in results])        
        smooth_x = lowess(align_x, align_plane, frac = 0.1, return_sorted = False)
        smooth_y = lowess(align_y, align_plane, frac = 0.1, return_sorted = False)
        print("Last dislocation (smoothed)", smooth_x[-1], smooth_y[-1])

        # make pandas dataframe
        return pandas.DataFrame({self.columns[0]: align_plane, \
                                 self.columns[1]: align_x, \
                                 self.columns[2]: align_y, \
                                 self.columns[3]: smooth_x, \
                                 self.columns[4]: smooth_y}, columns = self.columns)
