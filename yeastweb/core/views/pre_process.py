from django.shortcuts import  get_object_or_404, render
from django.http import HttpResponse
from core.models import UploadedImage 
# from ..models import Image 
from django.template.response import TemplateResponse
# Create your views here.
# chose function because https://spookylukey.github.io/django-views-the-right-way/context-data.html
def pre_process(request, uuid):
    image = get_object_or_404(UploadedImage, uuid=uuid)
    if request.method == "POST":
        preprocess_images(uuid, image)
        # form = UploadImageForm(request.POST, request.FILES)
        return HttpResponse("Preprocess completed")
    else:
        print("testing", uuid)
        print(image.file_location)
        # context = {'image' :}
        return TemplateResponse(request, "pre-process.html", {'image' : image})
# class HomePageView(ListView) :
#     model = Test
#     template_name = "home.html"


import numpy as np
from PIL import Image
import os
import skimage.exposure
import skimage.filters
from mrc import DVFile
from yeastweb.settings import MEDIA_ROOT
# def preprocess_images(inputdirectory, mask_dir, outputdirectory, outputfile, verbose = False, use_cache=True):
def preprocess_images(uuid, uploaded_image : UploadedImage):
    # if inputdirectory[-1] != "/":
    #     inputdirectory = inputdirectory + "/"
    # if outputdirectory[-1] != "/":
    #     outputdirectory = outputdirectory + "/"
    
    # Creates csv file and writes in first 2 columns ImageId and EncodedRLE
    # output = open(outputfile, "w")
    # output.write("ImageId, EncodedRLE" + "\n")
    # output.close()
    
    # for imagename in os.listdir(inputdirectory):
    # if '_PRJ' not in imagename:
        # continue
    # extspl = os.path.splitext(imagename)
    #check if there are .dv files and use them first
    # image = 0
    # print("ADAM",uploaded_image.file_location.open('r'))
    print("ADAM2", str(uploaded_image.file_location))
    print("ADAM3", MEDIA_ROOT)
    imagePath = os.path.join(MEDIA_ROOT, str(uploaded_image.file_location).replace("/", "\\"))
    print("ADAM4", imagePath)
    f = DVFile(imagePath)
    image = f.asarray()[0]
    # fileSize = os.path.getsize(uploaded_image.file_location)
    # if fileSize > 8230000:
        #File is a live cell imaging that has more than 4 images
    #     f = DVFile(uploaded_image)
    #     image = f.asarray()[0]
    # if extspl[1] == '.dv':
    #     f = DVFile(uploaded_image)
    #     image = f.asarray()[0]
        #if we don't have .dv files, see if there are tifs in the directory with the proper name structure

        # elif len(extspl) != 2 or extspl[1] != '.tif':  # ignore files that aren't tifs
        #     continue
        # else:
        #     image = np.array(Image.open(inputdirectory + imagename))
    # try:
        # if verbose:
        # print ("Preprocessing ", imagename)
        # existing_files = os.listdir(mask_dir)
        # if imagename in existing_files and use_cache:   #skip this if we have a mask already
            # continue
    outputdirectory = imagePath
    imagename = uploaded_image.name
    if len(image.shape) > 2:
        image = image[:, :, 0]
    height = image.shape[0]
    width = image.shape[1]

    # Preprocessing operations
    image = skimage.exposure.rescale_intensity(np.float32(image), out_range=(0, 1))
    image = np.round(image * 255).astype(np.uint8)        #convert to 8 bit
    image = np.expand_dims(image, axis=-1)
    rgbimage = np.tile(image, 3)                          #convert to RGB
    #rgbimage = skimage.filters.gaussian(rgbimage, sigma=(1,1))   # blur it first?
    imagename = imagename.split(".")[0]

    # if not os.path.exists(outputdirectory + imagename) or not use_cache:
    if not os.path.exists(outputdirectory + imagename):
        os.makedirs(outputdirectory + imagename)
        os.makedirs(outputdirectory + imagename + "/images/")
    rgbimage = Image.fromarray(rgbimage)
    rgbimage.save(outputdirectory + imagename + "/images/" + imagename + ".tif")

    # output = open(outputfile, "a")
    # output.write(imagename + ", " + str(height) + " " + str(width) + "\n")
    # output.close()
    print("FINISHED")
    # except IOError:
    #     pass

