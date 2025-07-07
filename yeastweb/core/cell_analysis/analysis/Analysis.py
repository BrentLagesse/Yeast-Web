from abc import abstractmethod
from ..image_processing import GrayImage

from core.models import CellStatistics

class Analysis:
    cp = None
    preprocessed_images = GrayImage()
    output_dir = None

    def __init__(self, cp:CellStatistics,image:GrayImage, output_dir):
        self.cp = cp
        self.preprocessed_images = image
        self.output_dir = output_dir

    @abstractmethod
    def calculate_statistics(self):
        pass