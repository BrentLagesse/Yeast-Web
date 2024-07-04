from django.http import HttpResponse
from django.shortcuts import redirect
from pathlib import Path
from PIL import Image
from yeastweb.settings import MEDIA_ROOT

import csv
import numpy as np
import os
import skimage.transform
import time

def rleToMask(rleString,height,width):
        rows,cols = height,width
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)

        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 255

        img = img.reshape(cols,rows)
        img = img.T

        return img

'''
Converts the compression rle files to images.

Input:
rlefile: csv file containing compressed masks from the segmentation algorithm
outputdirectory: directory to write images to
preprocessed_image_list: csv file containing list of images and their heights and widths
'''
def convert_to_image(request, uuid):
    # Assign variables that would normally be in the function header (in the original code)
    rescale = False
    scale_factor = 2
    verbose = False
    # Need to get the RLE file
    rle_file = Path(MEDIA_ROOT) / str(uuid) / "compressed_masks.csv"
    rle = csv.reader(open(rle_file), delimiter=',')
    rle = np.array([row for row in rle])[1:, :]

    # Need to get image list
    image_list_file = Path(MEDIA_ROOT) / str(uuid) / "preprocessed_images_list.csv"
    image_list = csv.reader(open(image_list_file), delimiter=',')
    image_list = np.array([row for row in image_list])[1:, :]

    # Need a directory to write the images we're about to create to
    outputdirectory = Path(MEDIA_ROOT) / str(uuid) / "output"
    os.makedirs(outputdirectory)

    files = np.unique(rle[:, 0])
    for f in files:
        if verbose:
            start_time = time.time()
        print ("Converting", f, "to mask...")

        list_index = np.where(image_list[:, 0] == f)[0][0]
        file_string = image_list[list_index, 1]

        size = file_string.split(" ")
        height = np.int(size[1])
        width = np.int(size[2])

        new_height = height
        new_width = width
        if rescale:
            new_height = height // scale_factor
            new_width = width // scale_factor

        image = np.zeros((new_height, new_width)).astype(np.float32)
        columns = np.where(rle[:, 0] == f)
        currobj = 1
        for i in columns[0]:
            currimg = rleToMask(rle[i, 1], new_height, new_width)
            currimg = currimg > 1
            image = image + (currimg * currobj)
            currobj = currobj + 1

        if rescale:
            image = skimage.transform.resize(image, output_shape = (height, width), order=0, preserve_range = True)

        # Creates an image and saves to \output
        # Depending on how we want to handle batches, we might not even need an output folder, we could potentially just save the mask under the uuid
        image = Image.fromarray(image)
        image.save(str(outputdirectory) + "\\mask.tif")

        if verbose:
            print ("Completed in", time.time() - start_time)

    return redirect(f'/image/{uuid}/segment/')
    
'''Converts masks to be ImageJ compatible'''
def convert_to_imagej(request, uuid):
    return