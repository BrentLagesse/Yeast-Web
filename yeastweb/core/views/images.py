from django.http import HttpResponse
from django.shortcuts import render
from ..forms import ImageForm
from ..models import Image, ImageForm
# Create your views here.
def upload_file(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        # form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # file = request.FILES['file']
            # instance = Image(cover=request.FILES["file"])
            # handle_uploaded_file(file)
            form.save()
            return HttpResponse("Image successfully uploaded")
    else:
        form = ImageForm()
    form = ImageForm()
    return render(request, 'form/uploadImage.html', {'form' : form})
    print("hello")
    
    
# https://docs.djangoproject.com/en/5.0/topics/http/file-uploads/
def handle_uploaded_file(file):
    with open("some/file/name.txt", "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)