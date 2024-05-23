Image_Processor README \
Author: Carter Savage

# Image Processor
This program is able to sharpen RGB and greyscale images. The program uses unsharp masking to sharpen and median filtering to remove noise.

## Setup
In order to run this program you must have Python3 installed.

In the terminal prompt simply type
```
sudo apt install python3
```


## Description
The code can be broken down into four main components are image sharpening for RGB and greyscale and noise removal for RGB and greyscale. The program uses threading and parallel programming in order to edit multiple pixles of the image at the same time. 
The program currently uses cpu cores but can very easily be changed on line 24 to 'gpu'.

### How To Use

To use the program have the file you want to open in the same directory and type 
```
python3 image.py <Algorithm (-s or -n)> <kernSize> <K value> <inFileName> <outFileName>
```
Algorithm: Type -s for sharpening and -n or noise removal.\
KernSize: Input the number of kernels you want to use by this program.\
K value: This value is used by the algorithms and affects how much the image is sharpened.\
inFileName: The name of the file to be edited.\
outFileName: The name of the edited file.
