# momomagick

A python toolbox for manipulating 2D/3D microscope multichannel time-lapse images saved in the TIFF or OME-TIFF format (X-Y-C-Z-T), especially those acquired with the [diSPIM light-sheet microscope](http://dispim.org) using [Micro-Manager](https://micro-manager.org/). The python scripts in this toolbox are used in our next paper, whose preprint will be available soon.

**momomagick** was named after [Micro-Manager](https://micro-manager.org/), the famous software for controlling microscope hardware, [ImageMagick](https://imagemagick.org/) and also after our hamster, **Momo**, who survived the COVID-19 pandemic with our family.

![momo.jpg](https://github.com/takushim/momomagick/raw/main/samples/momo.jpg)

Momo (2019-2020, RIP)

## Introduction

**momomagick** is a set of python scripts to handle 2D/3D images saved in the TIFF or OME-TIFF format. In this document, the basic usage of the scripts are described using the following sample images.
* time_lapse_2d.tif (in prep) - image cropping and registration
* time_lapse_3d.tif (in prep) - image cropping and registration
* single_channel_3d.tif (in prep) - deconvolution
* dual_channel_3d.tif (in prep) - fusion and deconvolution of dual-channel images

**Note:** Sample images and output images will be uploaded soon.

**Note:** Scripts in this toolkit were written for my TIF/OME-TIFF images acquired using diSPIM and MicroManager, which has the voxel size of 162.5 nm x 162.5 nm x 500 nm (X-Y-Z). The image resolution will fall back to these values when it is not available from the image file. **Keep your eyes on the log messages especially when you handle 3D images.**

## Getting Started

### Requirements

First of all, download and install the following programs.

* [`Python 3.12.1 or later`](https://www.python.org)
* [`Fiji (recommended)`](https://imagej.net/software/fiji/)
* [`Git (recommended)`](https://git-scm.com/)

Next, install the following packages to your Python environment using pip. You can install these packages directly to the Python system, but it is highly recommended to prepare [a virtual environment](https://docs.python.org/3/library/venv.html) and install packages on it. In this case, make sure to activate the virtual environment before running the program.

* `numpy`
* `pandas`
* `scipy`
* `Pillow`
* `tifffile`
* `ome-types`
* `NumpyEncoder`
* `statsmodels`
* `transforms3d` -- make sure to install "transforms3d", not transform3d
* `progressbar2` -- make sure to install "progressbar2", not progressbar

All of these libraries can be installed using `pip` by typing:
```
pip install numpy pandas scipy Pillow tifffile ome-types NumpyEncoder statsmodels transforms3d progressbar2
```

**Note:** Some scripts in this toolkit can work faster using nVidia GPU and cupy. For this purpose, setup the `CUDA environment` using guides (available for [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)). Then, install `cupy` using pip. GPU calculation is not activated by default. Use `-g 0` (the number may change when your workstation has multiple GPU boards) for GPU calculation.

### Installation

Download the zip file from my [GitHub repository](https://github.com/takushim/momomagick) and place all the files in an appropriate folder, for example, `C:\Users\[username]\momomagick` or `C:\Users\[username]\bin\momomagick`. Add the installed folder to the `PATH` environment variable.

**Note:** If you are using PowerShell, add `.PY` to the PATHEXT environment variable. Otherwise, Python will start in a separate window and finishes soon.

If [git](https://git-scm.com/) is installed, my git repository can be cloned using the following commend:
```
git clone https://github.com/takushim/momomagick.git
```

### Scripts in this toolkit

**Note:** Use `--help` option for the detailed usage.

This documents explains the usages of the following scripts.
* `mmcrop.py` - image cropping
* `mmregister.py`- registration of time-lapse images
* `mmdeconv.py` - deconvolution
* `mmfusion.py` - fusion and deconvolution of dual-channel images

Usages of these scripts are **not** covered by this documents.
* `mmlifetime.py` - calculate lifetime distribution or regression curves from the json file output from [momotrack](https://github.com/takushim/momotrack)
* `mmmark.py` - draw markers on images using the json file output from [momotrack](https://github.com/takushim/momotrack), especially used for images converted to 8 bit.
* `mmspotfilter.py` - filter the json file output from [momotrack](https://github.com/takushim/momotrack)
* `mmoverlay.py` - register two image stacks and output one multi-channel image

Algorithms are provided by the modules in the `mmtools` folder.
* `deconvolve.py` - deconvolution using cpu or gpu
* `draw.py` - draw various markers
* `gpuimage.py` - manipulate images using cpu or gpu
* `lifetime.py` - calculate lifetime distribution or regression curves
* `log.py` - logger
* `mmtiff.py` - obsolete and retained for compatibility
* `particles.py` - handle tracking records output from [momotrack](https://github.com/takushim/momotrack)
* `register.py` - register two images using cpu or gpu
* `stack.py` - load TIFF/OME-TIFF files  (optimized for MicroManager)
* `trackj.py` - obsolete and retained for compatibility

The `psf` folder contains images of PSF (point spread function) for the diSPIM microscope output generated using [`PSF Generator`](https://bigwww.epfl.ch/algorithms/psfgenerator/). You may want to check the `sh` folder for actual usages.

## Cropping

`mmcrop.py` simply crop 2D/3D images. The output images will be used by other scripts. The output filename is `XXX_crop.tif` unless otherwise specified. The `-R` option specifies the cropping area by `X Y Width Height`. Use `-z` and `-t` options to specify the range along the z and t axes. You can use `-c` option to select one channel. 

Usage:
```
mmcrop.py -R 0 0 256 256 time_lapse_2d.tif
mmcrop.py -R 0 0 256 256 time_lapse_3d.tif
```

## Registration

`mmregister.py` corrects sample drift during the time-lapse imaging. The output filename is `XXX_reg.tif` unless otherwise specified. The area used for registration can be specified by the `-R` option by `X Y Width Height`. The reference image can be specified by the `-r` option.

Image registration is preformed after applying a Hanning window. The following algorithms are available for registration. The algorithm for optimization should be `Powell` or `Nelder-Mead` in most cases (ignored for some registration algorithms, such as POC). GPU calculation is highly recommended for processing 3D images.

* `None` - do nothing
* `INTPOC` - pixel-order phase only correlation
* `POC` - subpixel-order phase only correlation
* `XY`- drift along the X-Y axes only
* `Drift` - drift only
* `Rigid` - drift + rotation (rigid body)
* `Rigid-Zoom` - drift + rotation + zoom
* `Full` - Full optimization of the affine matrix

Usage for CPU calculation:
```
mmregister.py -e POC -t Powell time_lapse_2d.tif
mmregister.py -e Rigid -t Powell time_lapse_3d.tif
```

Usage for GPU calculation:
```
mmregister.py -g 0 -e POC -t Powell time_lapse_2d.tif
mmregister.py -g 0 -e Rigid -t Powell time_lapse_3d.tif
```

## Deconvolution

`mmdeconv.py` deconvolves 3D images using a PSF function saved in a TIFF image. Files in the `PSF` folder are automatically selected and used unless otherwise specified. The output filename is `XXX_deconv.tif` by default. The algorithm is Richardson–Lucy. The `-i` option specifies the number of iteration.

**Note:** This script accepts 2D images, but deconvolution of 2D images is not recommended.

**Note:** The pixel size of PSF images are 165 nm x 165 nm x 165 nm or 165 nm x 165 nm x 500 nm. **Prepare a PSF image for your microscope setting and load it using the `-p` option**.

Usage (GPU calculation highly recommended):
```
mmregister.py -g 0 -i 10 -s single_channel_3d.tif
```

## Fusion of two channels for diSPIM imaging

The diSPIM microscope can image samples through volume scans in two directions perpendicular to each other. `mmfusion.py` selects two channels from a 3D image (`-m` and `-s`), rotate one channel (`-s`) by 90 or -90 degrees (`-r`) along the Y axis and fuses the two channels after registration (`-e Full`). Deconvolution will be performed when the `-i` option is specified. See the sample image and the output for the detail.

Usage (GPU calculation highly recommended):

```
mmfusion.py -g 0 -m 0 -s 1 -r 90 -e Full dual_channel_3d.tif
```

## Author

* **[Takushi Miyoshi](https://github.com/takushim)** - Assistant professor, Department of Biomedicine, Southern Illinois University School of Medicine

## License

This application is licensed under the MIT licence. All of the scripts in this toolkit were written by the author from scratch, but referring to the phase only correlation script, `poc.py`,  originally written by [Daisuke Kobayashi](https://github.com/daisukekobayashi/) and to the `diSPIMFusion` written in C/C++ by [Min Guo](https://github.com/eguomin).
