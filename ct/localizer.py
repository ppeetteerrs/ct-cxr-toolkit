from typing import Tuple

import cusignal as cs
import cv2 as cv
import numpy as np


def get_feature(img_2d: np.ndarray, percentage: int) -> np.ndarray:
    """
    Extracts the brightest N percent of pixels from image

    Args:
        img_2d (np.ndarray): 2D Image
        percentage (int): Percentage of pixels to extract

    Returns:
        np.ndarray: Extracted pixels
    """

    return np.where(img_2d < np.percentile(img_2d, 100 - percentage), 0, img_2d)


def align_features(
    sub_feature_2db: np.ndarray,
    base_feature_2db: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Calculates the corner coordinates of sub_feature_2db in
    base_feature_2db image by maximizing cross-correlation

    Args:
        sub_feature_2db (np.ndarray): Feature to be found
        base_feature_2db (np.ndarray): Image containing feature to be found

    Returns:
        Tuple[int, int, int, int]: row_start, row_end, col_start, col_end
    """

    # Calculate cross correlation
    sub_feature_2db = sub_feature_2db.astype(int)
    h, w = sub_feature_2db.shape
    sub_feature_2db[: h // 2, :] *= 3

    corr = cs.correlate2d(
        base_feature_2db.astype(int), sub_feature_2db, mode="same"
    ).get()

    # Get maximum cross correlation coordinates (sub_feature_2db relative to base_iamge)
    indices = np.unravel_index(corr.argmax(), base_feature_2db.shape)
    row_start, col_start = int(indices[0]), int(indices[1])
    row_start = max(row_start - sub_feature_2db.shape[0] // 2, 0)
    col_start = max(col_start - sub_feature_2db.shape[1] // 2, 0)
    row_end = row_start + sub_feature_2db.shape[0]
    col_end = col_start + sub_feature_2db.shape[1]
    return row_start, row_end, col_start, col_end


def crop_localizer(
    localizer: np.ndarray, crop: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops the localizer according to the corner coordinates and returns
    an RGB image showing the bounding box

    Args:
        localizer (np.ndarray): Localizer image
        crop (Tuple[int, int, int, int]): Corner coordinates

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cropped localizer, localizer with bounding box
    """

    row_start, row_end, col_start, col_end = crop
    cropped = localizer[row_start:row_end, col_start:col_end]
    rgb = cv.rectangle(
        cv.cvtColor(localizer, cv.COLOR_GRAY2RGB),
        (col_start, row_start),
        (col_end, row_end),
        (0, 255, 0),
        3,
    )
    return cropped, rgb


def get_localizer(
    ct_projection: np.ndarray, localizer: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align localizer scan with CT projection

    Args:
        ct_projection (np.ndarray): CT volume naiive projection
        localizer (np.ndarray): Localizer scan

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cropped localizer, localizer with bounding box, CT feature, localizer feature
    """

    ct_feature = get_feature(ct_projection, 40)
    localizer_feature = get_feature(localizer, 20)
    indices = align_features(ct_feature, localizer_feature)
    cropped, rgb = crop_localizer(localizer, indices)
    return cropped, rgb, ct_feature, localizer_feature
