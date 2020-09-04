#! /usr/bin/env python

# Copyright Daisuke Kobayashi (https://github.com/daisukekobayashi/phase-only-correlation)
# Modified by Takushi Miyoshi (2019) to work with python 3.6

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
import scipy, scipy.fftpack
from numpy import pi, sin, cos
from scipy.optimize import leastsq

def zero_padding(src, dstshape, pos = (0, 0)):
    y, x = int(pos[0]), int(pos[1])
    dst = numpy.zeros(dstshape)
    dst[y:src.shape[0] + y, x:src.shape[1] + x] = src
    return dst

def pocfunc_model(alpha, delta1, delta2, r, u):
    N1, N2 = r.shape
    V1, V2 = map(lambda x: 2 * x + 1, u)
    return lambda n1, n2: alpha / (N1 * N2) * sin((n1 + delta1) * V1 / N1 * pi) * sin((n2 + delta2) * V2 / N2 * pi)\
                                            / (sin((n1 + delta1) * pi / N1) * sin((n2 + delta2) * pi / N2))


def pocfunc(f, g, windowfunc = numpy.hanning, withlpf = True):
    m = numpy.floor(list(map(lambda x: x / 2.0, f.shape)))
    u = list(map(lambda x: x / 2.0, m))

    # hanning window
    hy = windowfunc(f.shape[0])
    hx = windowfunc(f.shape[1])
    hw = hy.reshape(hy.shape[0], 1) * hx
    f = f * hw
    g = g * hw

    # compute 2d fft
    F = scipy.fftpack.fft2(f)
    G = scipy.fftpack.fft2(g)
    G_ = numpy.conj(G)
    R = F * G_ / numpy.abs(F * G_)

    if withlpf == True:
        R = scipy.fftpack.fftshift(R)
        lpf = numpy.ones(list(map(lambda x: int(x + 1.0), m)))
        lpf = zero_padding(lpf, f.shape, u)
        R = R * lpf
        R = scipy.fftpack.fftshift(R)

    return scipy.fftpack.fftshift(numpy.real(scipy.fftpack.ifft2(R)))


def poc(f, g, fitting_shape = (9, 9)):
    # compute phase-only correlation
    #center = list(map(lambda x: x / 2.0, f.shape))
    m = numpy.floor(list(map(lambda x: x / 2.0, f.shape)))
    u = list(map(lambda x: x / 2.0, m))

    r = pocfunc(f, g)

    # least-square fitting
    max_pos = numpy.argmax(r)
    peak = (max_pos // f.shape[1], max_pos % f.shape[1])
    #max_peak = r[peak[0], peak[1]]

    mf = numpy.floor(list(map(lambda x: x / 2.0, fitting_shape))).astype(int)
    fitting_area = r[peak[0] - mf[0] : peak[0] + mf[0] + 1,\
                     peak[1] - mf[1] : peak[1] + mf[1] + 1]

    p0 = [0.5, -(peak[0] - m[0]) - 0.02, -(peak[1] - m[1]) - 0.02]
    y, x = numpy.mgrid[-mf[0]:mf[0] + 1, -mf[1]:mf[1] + 1]
    y = y + peak[0] - m[0]
    x = x + peak[1] - m[1]
    errorfunction = lambda p: numpy.ravel(pocfunc_model(p[0], p[1], p[2], r, u)(y, x) - fitting_area)
    plsq = leastsq(errorfunction, p0)
    return (plsq[0][0], plsq[0][1], plsq[0][2])
