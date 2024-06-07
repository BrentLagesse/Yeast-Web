from django.shortcuts import  get_object_or_404, render, get_list_or_404
from django.http import HttpResponse
from core.models import DVLayerTifPreview, UploadedImage
from django.template.response import TemplateResponse
from core.mrcnn.my_inference import predict_images
from core.mrcnn.preprocess_images import preprocess_images
import os
from .utils import tif_to_jpg
from yeastweb.settings import MEDIA_ROOT
from pathlib import Path
# chose function because https://spookylukey.github.io/django-views-the-right-way/context-data.html

def pre_process_step(request, uuid):
    """
        Shows screen of each dv's image array (should be 4 for each image)
        The user should confirm if the uploaded files are correct (name/content etc)
        After confirming, user clicks button to run models
    """
    # my_objects = list(DVLayerTifPreview.objects.filter(upload_image_uuid =uuid))
    preview_images = get_list_or_404(DVLayerTifPreview, uploaded_image_uuid=uuid)
    print("TEST" ,preview_images[0].file_location)
    if request.method == "POST":
        image = get_object_or_404(UploadedImage, uuid =uuid)
        output_directory = Path(MEDIA_ROOT) / str(uuid)
        preprocess_image_path, preprocessed_image_list_path = preprocess_images(uuid, image, output_directory)
        tif_to_jpg(Path(preprocess_image_path), Path(output_directory))
        rle_file = predict_images(preprocess_image_path, preprocessed_image_list_path, output_directory)
        return HttpResponse("Analysis completed")
    else:
        return TemplateResponse(request, "pre-process.html", {'images' : preview_images})



