from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from core.forms import UploadImageForm
from core.models import UploadedImage
import uuid

# Create your views here.
def upload_file(request):
    if request.method == "POST":
        # form = UploadImageForm(request.POST, request.FILES)
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            name = form.cleaned_data['name']
            file_location = request.FILES['file']
            image_uuid= uuid.uuid4()
            instance = UploadedImage(name=name, uuid=image_uuid, file_location=file_location )
            instance.save()
            # instance = Image(cover=request.FILES["file"])
            # handle_uploaded_file(file)
            # form.save()
            return redirect(f'/image/{image_uuid}/')
            return HttpResponse("Image successfully uploaded")
    else:
        form = UploadImageForm()
    form = UploadImageForm()
    return render(request, 'form/uploadImage.html', {'form' : form})
    print("hello")