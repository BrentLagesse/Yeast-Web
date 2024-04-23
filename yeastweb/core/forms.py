from core.models import Image
from django import forms


class ImageForm(forms.Form):
    name = forms.CharField(label="Picture Name", max_length=100)
    file = forms.FileField()

