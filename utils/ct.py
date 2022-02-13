from typing import Collection, Tuple, Union

import cv2 as cv
import numpy as np

from utils.cv import crop, min_max_normalize


def to_hu(val: int) -> int:
    """
    Convert 8-bit pixel value to Hounsfield unit

    Args:
        val (int): Pixel value

    Returns:
        int: Hounsfield unit
    """

    return int(val / 255 * 4095 - 1024)


def to_8bit(val: int) -> int:
    """
    Convert Hounsfield unit to 8-bit pixel value

    Args:
        val (int): Hounsfield unit

    Returns:
        int: Pixel value
    """

    return int((val + 1024) / 4095 * 255)


def project(
    img_3d: Collection[np.ndarray],
    equalized: bool = False,
    size: Union[int, Tuple[int, int]] = None,
    axis: int = 1,
) -> np.ndarray:
    """
    Simple projection of 3D volume to 2D view along an axis.
    Sum => (Crop) => Min-Max Normalize => (Equalize Hist)

    Args:
        img_3d (Collection[np.ndarray]): 3D image or list of 2D images
        axis (int, optional): Axis to sum along. Defaults to 1.

    Returns:
        np.ndarray: Projection
    """
    if not isinstance(img_3d, np.ndarray):
        img_3d = np.array(img_3d)

    img_2d = np.sum(img_3d, axis=axis)

    if size is not None:
        img_2d = crop(img_2d, size=size)

    img_2d = min_max_normalize(img_2d)

    if equalized:
        return cv.equalizeHist(img_2d)
    else:
        return img_2d
