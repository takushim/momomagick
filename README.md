# momoassist

A python toolbox for manipulating 2D/3D microscope images saved in the TIFF or OME-TIFF format, especially those acquired with the [diSPIM light-sheet microscope](http://dispim.org) using [Micro-Manager](https://micro-manager.org/). The python scripts in this toolbox are used in our next paper, whose preprint will be available soon.

**momoassist** was named after [Micro-Manager](https://micro-manager.org/), the famous software for controlling microscope hardware, and also after our hamster, **Momo**, who survived the COVID-19 pandemic with our family.

![momo.jpg](https://github.com/takushim/momoassist/raw/main/images/momo.jpg)

Momo (2019-2020, RIP)

## Introduction

**momoassist** is a set of python scripts to handle 2D/3D images saved in the TIFF or OME-TIFF format. In this document, the basic usage is described using the following images.
* A
* B
* C

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
* `transform3d`
* `progressbar2` -- make sure to install "progressbar2", not progressbar

All of these libraries can be installed using `pip` by typing:
```
pip install numpy pandas scipy Pillow tifffile ome-types NumpyEncoder statsmodels transform3d progressbar2
```

**Note:** Some scripts in this toolkit can work faster using nVidia GPU and cupy. For this purpose, setup the `CUDA environment` using guides (available for [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)). Then, install `cupy` using pip.

### Installation

Download the zip file from my [GitHub repository](https://github.com/takushim/momoassist) and place all the files in an appropriate folder, for example, `C:\Users\[username]\momoassist` or `C:\Users\[username]\bin\momoassist`. Add the installed folder to the `PATH` environment variable.

**Note:** If you are using PowerShell, add `.PY` to the PATHEXT environment variable. Otherwise, Python will start in a separate window and finishes soon.

If [git](https://git-scm.com/) is installed, my git repository can be cloned using the following commend:
```
git clone https://github.com/takushim/momoassist.git
```

### Scripts in this toolkit

SAMPLE

## Cropping

## Registration

## Deconvolution

## Fusion of two lightsheets (for diSPIM)

## Author

* **[Takushi Miyoshi](https://github.com/takushim)** - Assistant professor, Department of Biomedicine, Southern Illinois University School of Medicine

## License

This application is licensed under the MIT licence. All of the scripts in this toolkit were written by the author from scratch, but referring to the phase only correlation script, `poc.py`,  originally written by [Daisuke Kobayashi](https://github.com/daisukekobayashi/) and to the `diSPIMFusion` written in C/C++ by [Min Guo](https://github.com/eguomin).
