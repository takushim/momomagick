# momomagick
A python toolbox for manipulating 2D/3D microscope multichannel time-lapse images saved in the TIFF or OME-TIFF format (X-Y-C-Z-T), particularly those acquired using the [diSPIM light-sheet microscope](http://dispim.org) and [Micro-Manager](https://micro-manager.org/). This toolkit supports **accelaration using nVidia GPU boards**. The python scripts in this toolbox are used in our [single-molecule microscopy in live hair cell stereocilia preprinted at bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.04.590649v1).

**momomagick** was named after the famous software to control microscope hardware [Micro-Manager](https://micro-manager.org/), the famous software to manipulate image files [ImageMagick](https://imagemagick.org/) and the tiny friend **Momo**, who survived the COVID-19 pandemic with our family members.

![Momo (hamster)](https://github.com/takushim/momomagick/raw/main/samples/momo.jpg)

Momo (2020-2021, RIP)

## Introduction
**momomagick** is a set of python scripts to handle 2D/3D images saved in the TIFF or OME-TIFF format. This document describes the basic usage using the following sample images.
* time_lapse_2d.tif (in prep) - image cropping and registration
* time_lapse_3d.tif (in prep) - image cropping and registration
* single_channel_3d.tif (in prep) - deconvolution
* dual_channel_3d.tif (in prep) - fusion and deconvolution of dual-channel images

**Note:** Scripts in this toolkit are written for my TIF/OME-TIFF images acquired using diSPIM and MicroManager, which has the voxel size of 162.5 nm x 162.5 nm x 500 nm (X-Y-Z). The image resolution will fall back to these values when it is not specified in the image file. **Keep your eyes on the log messages especially when you handle 3D images.**

## Getting Started
### Setup the environment
**Note:** You don't have to use the following steps if you are familiar with the command-line user interface.

First, download and install the following programs.

* [Python 3.11.1 or later](https://www.python.org)
* [Git](https://git-scm.com/)
* [Fiji](https://imagej.net/software/fiji/) -  to view output images

**Note:** Check `Add python.exe to PATH` on the first page of the Python installer if you are not familiar with the command-line user interface. The Git installer will add Git commands to PATH automatically.

Next, install required packages using pip. You can install these packages directly to the Python system, but it is highly recommended to prepare [a virtual environment](https://docs.python.org/3/library/venv.html) to install packages.

* `numpy`
* `pandas`
* `scipy`
* `Pillow`
* `tifffile`
* `ome-types`
* `statsmodels`
* `transforms3d` -- make sure to install "transforms3d", not transform3d
* `progressbar2` -- make sure to install "progressbar2", not progressbar

Here are the steps to install required packages in a virtual environment named **momo**. You are making this environment in your home folder using PowerShell on Windows.

1. Make sure that `*.py` files are associated with python.exe using File Explorer. 
2. Open PowerShell.
3. Move to a folder where you want to place files for the environment (use `cd` command).
```
cd $HOME
```
4. Make a virtual enviconment named **momo**. Enter the full path to python.exe instead of python if the path to python.exe is not in the `PATH` environment variable. 
```
python -m venv momo
```
5. Activate the **momo** virtual environment.
```
momo/Scripts/Activate.ps1
```
6. Update pip (optional). `pip` is automatically added to `PATH`.
```
py -m pip install pip -U
```
7. Install required packages.
```
pip install numpy pandas scipy Pillow tifffile ome-types statsmodels transforms3d progressbar2
```

### Setup the CUDA environment (optional)
Some scripts can run faster using nVidia GPU and cupy.
1. Setup the CUDA environment using guides for [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
2. Hit a pip command to install the `cupy` package. Install cupy-cuda11x for CUDA version 11.
```
pip install cupy-cuda12x
```
3. Use the Python script with an option `-g 0`. The number may change when your PC has multiple GPU boards.

### Download scripts
Here are the steps to install scripts in your home folder using PowerShell on Windows.

1. Open PowerShell.
2. Move to your home folder.
```
cd $HOME
```
3. Clone my [GitHub repository](https://github.com/takushim/momomagick).
```
git clone https://github.com/takushim/momomagick.git
```

### Before running scripts
You will have to update environmental variables before running scripts.
1. Open PowerShell and move to the folder storing your images using the `cd` command. You can also open PowerShell in the current folder using File Explorer.
2. Set the environmental variables. You can set environmental variables permanently using Windows Settings or Profile.ps1.
```
$env:PATHEXT = "${env:PATHEXT}:.PY"
```
```
$env:PATH = "${env:PATH};${HOME}/momomagick"
```
3. Run scripts **without closing the current PowerShell window**.

### Use scripts

This documents explains the usages of the following scripts. Use `--help` option for the detailed usage.
* `mmcrop.py` - Image cropping
* `mmregister.py`- Registration of time-lapse images
* `mmdeconv.py` - Deconvolution
* `mmfusion.py` - Fusion and deconvolution of dual-channel images

Usages of these scripts are **not** covered by this documents.
* `mmlifetime.py` - Calculate lifetime distribution or regression curves from the json file output from [momotrack](https://github.com/takushim/momotrack)
* `mmmark.py` - Draw markers on images using the json file output from [momotrack](https://github.com/takushim/momotrack). Usually markers are drawn on images converted to the 8-bit format.
* `mmspotfilter.py` - Filter the json file output from [momotrack](https://github.com/takushim/momotrack)
* `mmoverlay.py` - Register two image stacks and output one multi-channel image
* `mmswapaxis.py` - Swap T and Z axes of TIFF files output from ImageJ.

Algorithms are provided by the modules in the `mmtools` folder.
* `register.py` - Register two images using cpu or gpu
* `deconvolve.py` - Deconvolve images using cpu or gpu
* `stack.py` - Load TIFF/OME-TIFF files  (optimized for MicroManager)
* `gpuimage.py` - Manipulate images using cpu or gpu
* `draw.py` - Draw various markers
* `lifetime.py` - Calculate lifetime distribution or regression curves
* `particles.py` - Handle tracking records output from [momotrack](https://github.com/takushim/momotrack)
* `npencode.py` - Output numpy instances to json. Implemented referring to [NumpyEncoder](https://github.com/hmallen/numpyencoder).
* `log.py` - Logger
* `mmtiff.py` - Obsolete and retained for compatibility
* `trackj.py` - Obsolete and retained for compatibility

The `psf` folder contains images of PSF (point spread function) for the diSPIM microscope output generated using [`PSF Generator`](https://bigwww.epfl.ch/algorithms/psfgenerator/). You may want to check the `sh` folder for the automation.

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
