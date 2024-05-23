from django.shortcuts import  get_object_or_404, render
from django.http import HttpResponse
from core.models import UploadedImage 
from django.template.response import TemplateResponse
from .predict_images import predict_images
from core.mrcnn.preprocess_images import preprocess_images
import os
from yeastweb.settings import MEDIA_ROOT
# chose function because https://spookylukey.github.io/django-views-the-right-way/context-data.html

def pre_process(request, uuid):
    image = get_object_or_404(UploadedImage, uuid=uuid)

    if request.method == "POST":
        output_directory = os.path.join(MEDIA_ROOT, str(uuid))
        preprocess_image_path, preprocessed_image_list_path = preprocess_images(uuid, image, output_directory)
        rle_file = predict_images(preprocess_image_path, preprocessed_image_list_path, output_directory)
        return HttpResponse("Preprocess completed")
    else:
        print(image.file_location)
        # context = {'image' :}
        return TemplateResponse(request, "pre-process.html", {'image' : image})
# class HomePageView(ListView) :
#     model = Test
#     template_name = "home.html"




