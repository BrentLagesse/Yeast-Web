import cv2, os, csv
import numpy as np
from core.models import Contour


def calculate_nucleus_intensity(cp,gray_GFP,best_contour,orig_gray_GFP_no_bg,mcherry_line_pts,output_dir):
    """
    This function calculate the nucleus intensity within a green image
    :param gray: Gray scale of green image
    """
    mask_contour = np.zeros(gray_GFP.shape, np.uint8)
    cv2.fillPoly(mask_contour, [best_contour], 255)
    pts_contour = np.transpose(np.nonzero(mask_contour))

    # Build the expected outline filename:
    # cp.image_name is set (in the get_or_create for CellStatistics) as DV_Name + '.dv',
    # so taking os.path.splitext(cp.image_name)[0] gives the full DV name (e.g. "M3850_001_PRJ")
    outline_filename = os.path.splitext(cp.image_name)[0] + '-' + str(cp.cell_id) + '.outline'

    # The outline files are stored in the "output" folder (not in a "masks" folder)
    mask_file_path = os.path.join(output_dir, 'output', outline_filename)

    with open(mask_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        border_cells = []
        for row in csvreader:
            border_cells.append([int(row[0]), int(row[1])])

    # Calculate nucleus intensity inside the best_contour
    intensity_sum = 0
    for p in pts_contour:
        intensity_sum += orig_gray_GFP_no_bg[p[0]][p[1]]

    # Cast to Python int before saving into the JSON field
    cp.nucleus_intensity[Contour.CONTOUR.name] = int(intensity_sum)
    cp.nucleus_total_points = len(pts_contour)  # This is usually a Python int already

    cp.nucleus_intensity_sum = float(intensity_sum)

    # Calculate cell intensity from the "border_cells" list
    cell_intensity_sum = 0
    for p in border_cells:
        cell_intensity_sum += orig_gray_GFP_no_bg[p[0]][p[1]]

    # Ensure that the JSON field gets a Python int
    cp.cell_intensity = int(cell_intensity_sum)
    cp.cell_total_points = len(border_cells)

    cp.cellular_intensity_sum = float(cell_intensity_sum)

    # Calculate mCherry line intensity
    mcherry_line_intensity_sum = 0
    for p in mcherry_line_pts:
        mcherry_line_intensity_sum += orig_gray_GFP_no_bg[p[0]][p[1]]

    # Again, cast to a Python int
    cp.mcherry_line_gfp_intensity = int(mcherry_line_intensity_sum)

    cp.line_gfp_intensity = float(mcherry_line_intensity_sum)
