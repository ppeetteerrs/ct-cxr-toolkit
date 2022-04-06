from typing import Tuple

import cv2 as cv
import numpy as np
from utils.cv import CONTOUR_IDX, denoise


def get_bone_mask(
    img_2d: np.ndarray, bone_threshold: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts bones from CT slice

    Args:
        img_2d (np.ndarray): CT slice
        bone_threshold (int): Lower threshold for bones

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: bones, threshold, filled contours
    """

    # Apply threshold to extract bone-like structures
    threshold = np.where(img_2d < bone_threshold, 0, 255).astype(np.uint8)

    # Fill all contours (i.e. fill in bone marrow?!)
    contours = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[
        CONTOUR_IDX
    ]
    contour_img = np.zeros_like(threshold)

    if len(contours) > 0:
        cv.drawContours(contour_img, contours, -1, 255, cv.FILLED)

    # Denoise the image (remove small dots)
    bones = denoise(contour_img, kernel_size=2)

    return bones, threshold, contour_img
