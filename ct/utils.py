from typing import Collection, List, Tuple, Union

import cv2 as cv
import numpy as np
import SimpleITK as sitk

from utils.cv import crop, min_max_normalize


def read_dcm(
    paths: List[str], spacing: Tuple[int, int, int] = (1, 1, 1)
) -> Tuple[sitk.Image, np.ndarray]:
    """
    Reads a DICOM series and returns a resampled output
    Hounsfield Unit range: [-1024, 3071]
    """

    # Read image using sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    image: sitk.Image = reader.Execute()

    # Resample image to new pixel spacing
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [
        int(i * j / k) for i, j, k in zip(original_size, original_spacing, spacing)
    ]
    image = sitk.Resample(
        image1=image,
        size=new_size,
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=image.GetOrigin(),
        outputSpacing=spacing,
        outputDirection=image.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=image.GetPixelID(),
    )
    np_img = sitk.GetArrayFromImage(image)
    return image, np_img


def normalize_hu(img: np.ndarray) -> np.ndarray:
    """
    Normalizes a Hounsfield image to an 8-bit image.
    """
    return ((img + 1024) / 4095 * 255).astype(np.uint8)


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
