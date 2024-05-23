# Carter Savage 1103661
import sys
import warp as wp
import numpy as np
from PIL import Image, ImageFilter

# Input checking
if(sys.argv[1] == "-s"):
    sharp = 1
elif(sys.argv[1] == "-n"):
    sharp = 0
else:
    print("Error incorrect arguments Eg:python3 a4.py algType kernSize param inFileName outFileName")
    exit(1)
if(((int(sys.argv[2]) % 2) == 0 ) or (int(sys.argv[2]) <= 0)):
    print("Error kernel size must be odd and greater then 0")
    exit(1)
if(len(sys.argv) != 6):
    print("Error incorrect number of arguments Eg:python3 a4.py algType kernSize param inFileName outFileName")
    exit(1)
image = Image.open(sys.argv[4])
wp.init()

device = "cpu"
# need these for Kernels that do not have them
x = wp.constant(1)
y = wp.constant(1) 
kVal = wp.constant(1.0)
@wp.kernel
def greyscaleSharp(imgArr: wp.array(dtype=float, ndim=2),
            output: wp.array(dtype=float, ndim=2)):
    # thread index
    i, j = wp.tid()
    temp = float(0)
    for l in range (int(-kernSize/2), int(kernSize/2)):
        for m in range (int(-kernSize/2), int(kernSize/2)):
            if((i + l >= x or i + l < 0) and (j + m >= y or j + m < 0) ): # X and Y are out of bounds          
                temp += imgArr[i - l, j - m]
            elif((i + l >= x or i + l < 0)): # Just X is out of bounds          
                temp += imgArr[i - l, j + m]
            elif((j + m >= y or j + m < 0)): # Just Y is out of bounds          
                temp += imgArr[i + l, j - m]
            else: # X and Y are in bounds          
                temp += imgArr[i + l, j + m]
    temp = temp/float((kernSize * kernSize))
    temp = imgArr[i, j] - temp
    output[i, j] = imgArr[i, j] + (kVal * temp)
    if output[i, j] > 255.0:
        output[i, j] = 255.0
    elif output[i, j] < 0.0:
        output[i, j] = 0.0


@wp.kernel
def RGBASharp(imgArr: wp.array(dtype=float, ndim=3),
           output: wp.array(dtype=float, ndim=3)):
    # thread index
    i, j, l = wp.tid()
    temp = float(0)
    if l == 3:
        output[i, j, l] = imgArr[i, j, l]
    else:
        # compute distance of each point from origin
        for n in range (int(-kernSize/2), int(kernSize/2)):
            for m in range (int(-kernSize/2), int(kernSize/2)):
                if((i + n >= x or i + n < 0) and (j + m >= y or j + m < 0) ): # X and Y are out of bounds          
                    temp += imgArr[i - l, j - m, l]
                elif((i + n >= x or i + n < 0)): # Just X is out of bounds          
                    temp += imgArr[i - n, j - m, l]
                elif((j + m >= y or j + m < 0)): # Just Y is out of bounds          
                    temp += imgArr[i - n, j - m, l]
                else: # X and Y are in bounds          
                    temp += imgArr[i - n, j - m, l]
        temp = temp/float((kernSize * kernSize))
        temp = imgArr[i, j, l] - temp
        output[i, j, l] = imgArr[i, j, l] + (kVal * temp)
        if output[i, j, l] > 255.0:
            output[i, j, l] = 255.0
        elif output[i, j, l] < 0.0:
            output[i, j, l] = 0.0

@wp.kernel
def greyscaleNoise(imgArr: wp.array(dtype=float, ndim=2),
            values: wp.array(dtype=float, ndim=2),
            output: wp.array(dtype=float, ndim=2)):
    # thread index
    index = int(0)
    i, j = wp.tid()
    for l in range (int(-kernSize/2), int(kernSize/2)):
        for m in range (int(-kernSize/2), int(kernSize/2)):
            if((i + l >= x or i + l < 0) and (j + m >= y or j + m < 0) ): # X and Y are out of bounds          
                values[0][index] = imgArr[i - l, j - m]
            elif((i + l >= x or i + l < 0)): # Just X is out of bounds          
                values[0][index] = imgArr[i - l, j + m]
            elif((j + m >= y or j + m < 0)): # Just Y is out of bounds          
                values[0][index] = imgArr[i + l, j - m]
            else: # X and Y are in bounds          
                values[0][index] = imgArr[i + l, j + m]          
            index += int(1)
    # Sorting
    for l in range(0, index - 1):
        smallest_id = l
        for value in range(l + 1, index):
            if (values[0][value] > values[0][smallest_id]):
                smallest_id = value
        tempVal = values[0][l]
        values[0][l] = values[0][smallest_id]
        values[0][smallest_id] = tempVal
    output[i, j] = values[0][index/2]

@wp.kernel
def RGBAnoise(imgArr: wp.array(dtype=float, ndim=3),
            values: wp.array(dtype=float, ndim=3),
            output: wp.array(dtype=float, ndim=3)):
    # thread index
    index = int(0)
    i, j, k = wp.tid()
    if k == 3:
        output[i, j, k] = imgArr[i, j, k]
    else:
        for l in range (int(-kernSize/2), int(kernSize/2)):
            for m in range (int(-kernSize/2), int(kernSize/2)):
                if((i + l >= x or i + l < 0) and (j + m >= y or j + m < 0) ): # X and Y are out of bounds          
                    values[0][index][0] = imgArr[i - l, j - m, k]
                elif((i + l >= x or i + l < 0)): # Just X is out of bounds          
                    values[0][index][0] = imgArr[i - l, j + m, k]
                elif((j + m >= y or j + m < 0)): # Just Y is out of bounds          
                    values[0][index][0] = imgArr[i + l, j - m, k]
                else: # X and Y are in bounds          
                    values[0][index][0] = imgArr[i + l, j + m, k]          
            index += int(1)          
        # Sorting
        for l in range(0, index - 1):
            smallest_id = l
            for value in range(l + 1, index):
                if (values[0][value][0] > values[0][smallest_id][0]):
                    smallest_id = value
            tempVal = values[0][l][0]
            values[0][l][0] = values[0][smallest_id][0]
            values[0][smallest_id][0] = tempVal
        output[i, j, k] = values[0][index/2][0]


# Grey sharp
if(image.mode == "L" and sharp == 1):
    numpyArr = np.asarray(image, dtype='float32')
    imgArr = wp.array(numpyArr, dtype=float, device=device)
    output = wp.zeros(numpyArr.shape, dtype=float, device=device)
    kVal = wp.constant(float(sys.argv[3]))
    kernSize = wp.constant(int(sys.argv[2]))
    x = wp.constant(int(numpyArr.shape[0]))
    y = wp.constant(int(numpyArr.shape[1]))
    # launch kernel
    wp.launch(kernel=greyscaleSharp,
            dim=numpyArr.shape,
            inputs=[imgArr, output],
            device=device)
    numpyOutArr = output.numpy()
    imageOut = Image.fromarray(np.uint8(numpyOutArr))
    imageOut.save(sys.argv[5])

# RGB sharp 
elif(image.mode == "RGBA" or image.mode == "RGB" and sharp == 1):
    numpyArr = np.asarray(image, dtype='float32')
    imgArr = wp.array(numpyArr, dtype=float, device=device)
    output = wp.zeros(numpyArr.shape, dtype=float, device=device)
    kVal = wp.constant(float(sys.argv[3]))
    kernSize = wp.constant(int(sys.argv[2]))
    x = wp.constant(int(numpyArr.shape[0]))
    y = wp.constant(int(numpyArr.shape[1]))
    # launch kernel
    wp.launch(kernel=RGBASharp,
            dim=numpyArr.shape,
            inputs=[imgArr, output],
            device=device)
    numpyOutArr = output.numpy()
    imageOut = Image.fromarray(np.uint8(numpyOutArr))
    imageOut.save(sys.argv[5])

# Grey noise 
elif(image.mode == "L" and sharp == 0):
    numpyArr = np.asarray(image, dtype='float32')
    imgArr = wp.array(numpyArr, dtype=float, device=device)
    output = wp.zeros(numpyArr.shape, dtype=float, device=device)
    x = wp.constant(int(numpyArr.shape[0]))
    y = wp.constant(int(numpyArr.shape[1]))
    kernSize = wp.constant(int(sys.argv[2]))
    values = wp.zeros(numpyArr.shape, dtype=float, device=device)
    # launch kernel
    wp.launch(kernel=greyscaleNoise,
            dim=numpyArr.shape,
            inputs=[imgArr, values, output],
            device=device)
    numpyOutArr = output.numpy()
    imageOut = Image.fromarray(np.uint8(numpyOutArr))
    imageOut.save(sys.argv[5])


# RGB Noise 
elif(image.mode == "RGBA" or image.mode == "RGB" and sharp == 0):
    numpyArr = np.asarray(image, dtype='float32')
    imgArr = wp.array(numpyArr, dtype=float, device=device)
    output = wp.zeros(numpyArr.shape, dtype=float, device=device)
    x = wp.constant(int(numpyArr.shape[0]))
    y = wp.constant(int(numpyArr.shape[1]))
    kernSize = wp.constant(int(sys.argv[2]))
    values = wp.zeros(numpyArr.shape, dtype=float, device=device)
    # launch kernel
    wp.launch(kernel=RGBAnoise,
            dim=numpyArr.shape,
            inputs=[imgArr, values, output],
            device=device)
    numpyOutArr = output.numpy()
    imageOut = Image.fromarray(np.uint8(numpyOutArr))
    imageOut.save(sys.argv[5])
