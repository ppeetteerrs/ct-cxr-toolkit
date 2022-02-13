from typing import List, Tuple

import numpy as np
import SimpleITK as sitk


def read_dcm(
    paths: List[str], spacing: Tuple[int, int, int] = (1, 1, 1)
) -> Tuple[sitk.Image, np.ndarray]:
    """
    Reads a DICOM series and returns a resampled output with blank slices removed.
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
