from typing import Sequence, Tuple, Union

import cv2 as cv
import numpy as np

CONTOUR_IDX = 0 if int(cv.__version__[0]) >= 4 else 1


def mask(img: np.ndarray, mask: np.ndarray, val: int = 0) -> np.ndarray:
    return np.where(mask, img, val)


def circle_mask(vol_3d: np.ndarray) -> np.ndarray:
    N = vol_3d.shape[1]
    center = N // 2
    for i in range(N):
        for j in range(N):
            if ((i - center) ** 2 + (j - center) ** 2) > (N // 2 - 1) ** 2:
                vol_3d[:, i, j] = 0
    return vol_3d


def remove_border(img: np.ndarray, tol=50) -> np.ndarray:
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def crop(img: np.ndarray, size: Union[Tuple[int, int], int] = None) -> np.ndarray:
    """
    Center-crops and resizes an image

    Args:
        img (np.ndarray): Input image
        size (Union[Tuple[int, int], int], optional): Height, width. Defaults to None.

    Returns:
        np.ndarray: Output image
    """
    curr_h, curr_w = new_h, new_w = img.shape

    # Calculate new height and width
    if size is not None:
        if isinstance(size, int):
            new_h = new_w = size
        else:
            new_h, new_w = size

    # Height and width to crop to
    crop_h = int(min(curr_h, curr_w / new_w * new_h))
    crop_w = int(crop_h / new_h * new_w)

    row_start = (curr_h - crop_h) // 2
    col_start = (curr_w - crop_w) // 2

    img = img.astype(np.float32)
    return cv.resize(
        img[row_start : row_start + crop_h, col_start : col_start + crop_w],
        (new_w, new_h),
    )


def concat(imgs: Sequence[np.ndarray], axis=1) -> np.ndarray:
    return np.concatenate(imgs, axis=axis)


def denoise(
    img: np.ndarray,
    kernel_size: int = 3,
    erode_iter: int = 1,
    dilate_iter: int = 1,
    total_iter: int = 1,
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for _ in range(total_iter):
        img = cv.erode(img, kernel, iterations=erode_iter)
        img = cv.dilate(img, kernel, iterations=dilate_iter)
    return img


def renoise(
    img: np.ndarray,
    kernel_size: int = 3,
    erode_iter: int = 1,
    dilate_iter: int = 1,
    total_iter: int = 1,
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for _ in range(total_iter):
        img = cv.dilate(img, kernel, iterations=dilate_iter)
        img = cv.erode(img, kernel, iterations=erode_iter)
    return img


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    lower = img.min()
    upper = np.percentile(img, 97)
    img[img > upper] = upper
    return ((img - lower) / (upper - lower) * 255).astype(np.uint8)
