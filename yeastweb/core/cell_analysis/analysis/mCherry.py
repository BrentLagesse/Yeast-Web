import math, cv2
import numpy as np
from ..utils.contour_helper import get_contour_center
from ..utils.image_helper import calculate_intensity_mask,create_circular_mask
from ..image_processing import GrayImage

def mcherry_line_calculation(cp, contours_mcherry,best_mcherry_contours, mcherry_line_width_input,original_image,preprocessed_images:GrayImage):
    """
    This function calculates the mCherry line distance of a cell pair.
    :param cp: CellStatistics object
    :param contours_mcherry: List of contours in mCherry
    :param best_mcherry_contours: Index of best contours in mCherry
    :return: Distance between centers of cell pair
    """
    mcherry_line_pts = []
    if len(best_mcherry_contours) == 2:
        # choose two best contour
        c1 = contours_mcherry[0][best_mcherry_contours[0]]
        c2 = contours_mcherry[0][best_mcherry_contours[1]]

        # getting 2 centers of contours
        try:
            centers = get_contour_center([c1, c2])
            # distance between 2 contour
            d = math.dist(centers[0],centers[1])
            # Directly assign to cp.red_dot_distance (instead of cp.set_red_dot_distance(d))
            cp.red_dot_distance = d
            cp.distance = float(d)

            c1x, c1y = centers[0]
            c2x, c2y = centers[1]

            # Use a 3-channel white color tuple:
            cv2.line(original_image, (c1x, c1y), (c2x, c2y), (255, 255, 255), int(mcherry_line_width_input))
            gray_mCherry = preprocessed_images.get_image('gray_mcherry')
            mcherry_line_mask = np.zeros(gray_mCherry.shape, np.uint8)
            cv2.line(mcherry_line_mask, (c1x, c1y), (c2x, c2y), 255, int(mcherry_line_width_input))
            mcherry_line_pts = np.transpose(np.nonzero(mcherry_line_mask))

            return mcherry_line_pts

        except ZeroDivisionError:
            print("can't find contours")
            return []
    else:
        return []

def identify_red_signal(red_image, intensity):
    """
    Identify red signal from mCherry image
    :param red_image: Gray scale of mCherry image
    :param intensity: Threshold for detection
    :return: list of red dot's center coordinates
    """
    red_dot = []

    _, thresh = cv2.threshold(red_image,intensity,255,cv2.THRESH_BINARY) # zeroing the value under the thresh hold
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 1000: #TODO: Make area adjustable
            red_dot.append(get_contour_center([contour])[0])

    return red_dot

def calculate_red_green_intensity(preprocessed_images:GrayImage):
    """
    :param preprocessed_images: GrayImage object
    :return: ratio between red and green intensity
    """
    mcherry_gray = preprocessed_images.get_image('gray_mcherry')
    GFP_gray = preprocessed_images.get_image('GFP')

    red_dot = identify_red_signal(mcherry_gray, 10) # identify red signal from the mCherry
    ratio = 0
    for i in red_dot:
        mask = create_circular_mask(mcherry_gray.shape, i, 10) # draw a contour around red signal TODO: make the radius configurable
        red_intensity = calculate_intensity_mask(mcherry_gray, mask)
        green_intensity = calculate_intensity_mask(GFP_gray, mask)

        ratio = green_intensity / red_intensity if red_intensity != 0 else 0
    return ratio