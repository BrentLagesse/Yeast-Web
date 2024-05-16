import uuid
from django.db import models
# https://docs.djangoproject.com/en/5.0/topics/forms/modelforms/#django.forms.ModelForm
from django.forms import ModelForm



def picture_path(instance, filename):
    uuid = instance.uuid
    return f'images/{uuid}/uploaded-image.dv'

class UploadedImage(models.Model):
    name = models.TextField()
    uuid = models.UUIDField(primary_key=False, default=uuid.uuid4, editable=False)
    file_location = models.FileField(upload_to=picture_path)
    def __str__(self):
        return self.name


# class Contour(Enum):
#     CONTOUR = 0
#     CONVEX = 1
#     CIRCLE = 2

# class CellPair:
#     def __init__(self, image_name, id):
#         # https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html
#         self.is_correct = True # if is affine and is separable
#         self.image_name = image_name # 20_1212_M1914_001_R3D_REF.tif
#         self.id = id # number of cells undergoing mitosis
#         self.nuclei_count = 1 
#         self.red_dot_count = 1
#         self.gfp_dot_count = 0
#         self.red_dot_distance = 0
#         self.gfp_red_dot_distance = 0
#         self.cyan_dot_count = 1
#         self.green_dot_count = 1
#         self.ground_truth = False
#         self.nucleus_intensity = {}
#         self.nucleus_total_points = 0
#         self.cell_intensity = {}
#         self.cell_total_points = 0
#         self.ignored = False
#         self.mcherry_line_gfp_intensity = 0
#         self.gfp_line_gfp_intensity = 0
#         self.properties = dict()
        




