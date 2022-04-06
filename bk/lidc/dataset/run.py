from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from lidc.dataset.body_mask import get_body_mask
from lidc.dataset.bone_mask import get_bone_mask
from lidc.dataset.localizer import align_features, crop_localizer, get_feature
from tqdm import tqdm
from utils.ct import project, to_8bit
from utils.cv import concat, crop, min_max_normalize
from utils.dicom import normalize_hu, read_dcm
from utils.lmdb import LMDBImageWriter, covid_ct_indexer
from utils.utils import save_img, track

"""
Convention:
    _3d: 3D volume or list of 2D images
    _2d: 2D image (original resolution)
    _2db: 2D binary mask
    _2dN: NxN 2D image
    _2dNxM: NxM 2D image
"""


def make_circle(N: int = 350) -> np.ndarray:
    circle = np.zeros((N, N))
    center = N // 2

    for i in range(N):
        for j in range(N):
            if ((i - center) ** 2 + (j - center) ** 2) > (N // 2 - 1) ** 2:
                circle[i][j] = 1
    return circle


def circle_mask(vol_3d: np.ndarray) -> np.ndarray:
    N = vol_3d.shape[1]
    center = N // 2
    for i in range(N):
        for j in range(N):
            if ((i - center) ** 2 + (j - center) ** 2) > (N // 2 - 1) ** 2:
                vol_3d[:, i, j] = 0
    return vol_3d


def process_slice(lung_2d: np.ndarray):
    # Extract patient body from bed
    body_2db, body_thres_2db, body_contour_2db = get_body_mask(
        lung_2d, body_threshold=35
    )
    body_2d = np.where(body_2db, lung_2d, 0).astype(np.uint8)

    # Extract bone mask
    bones_2db, bones_thres_2db, bones_contour_2db = get_bone_mask(body_2d, to_8bit(250))

    soft_2d = np.where(bones_2db, 0, body_2d)

    return (
        body_2d,
        soft_2d,
        bones_2db,
        body_thres_2db,
        body_contour_2db,
        bones_thres_2db,
        bones_contour_2db,
    )


def process_lung(lung_3d: np.ndarray):
    # Get localizer
    lung_frontal_2d = project(lung_3d)
    (
        body_3d,
        soft_3d,
        bones_3db,
        body_thres_3db,
        body_contour_3db,
        bones_thres_3db,
        bones_contour_3db,
    ) = zip(*[process_slice(lung_2d) for lung_2d in lung_3d])

    raw_2d = crop(lung_frontal_2d, size=256)
    body_2d = project(body_3d, size=256)
    soft_2d = project(soft_3d, size=256)
    bones_2d = project(bones_3db, size=256, equalized=True)

    return raw_2d, body_2d, soft_2d, bones_2d


def lidc_dataset(output_dir: Path):
    df = pd.read_pickle(output_dir / "metadata.pkl")
    items: List[Tuple[str, ...]] = df["lung"].values.tolist()

    (output_dir / "lung").mkdir(parents=True, exist_ok=True)
    (output_dir / "soft").mkdir(parents=True, exist_ok=True)
    (output_dir / "bones").mkdir(parents=True, exist_ok=True)
    samples: List[np.ndarray] = []

    for idx, lung_paths in track(list(enumerate(items)), description="Processing"):
        _, lung_3d = read_dcm(list(lung_paths))
        lung_3d = circle_mask(normalize_hu(lung_3d))

        # Get lung, localizer, body, bones, drr
        raw_2d, body_2d, soft_2d, bones_2d = process_lung(lung_3d)

        # Save images to different places
        save_img(body_2d, output_dir / f"lung/{str(idx).zfill(6)}.png")
        save_img(soft_2d, output_dir / f"soft/{str(idx).zfill(6)}.png")
        save_img(bones_2d, output_dir / f"bones/{str(idx).zfill(6)}.png")

        samples.append(concat((raw_2d, body_2d, soft_2d, bones_2d), axis=1))

    save_img(
        concat(
            samples,
            axis=0,
        ),
        output_dir / "samples.png",
    )


def to_lmdb(output_dir: Path):
    n_items = pd.read_pickle(output_dir / "metadata.pkl").shape[0]

    lmdb_dir = output_dir / "lmdb"
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    writer = LMDBImageWriter(lmdb_dir, covid_ct_indexer)

    for idx in tqdm(range(n_items)):
        lung = cv.imread(
            str(output_dir / f"lung/{str(idx).zfill(6)}.png"), cv.IMREAD_GRAYSCALE
        )
        loc = cv.imread(
            str(output_dir / f"localizer/{str(idx).zfill(6)}.png"), cv.IMREAD_GRAYSCALE
        )
        drr = cv.imread(
            str(output_dir / f"drr/{str(idx).zfill(6)}.png"), cv.IMREAD_GRAYSCALE
        )
        bones = cv.imread(
            str(output_dir / f"bones/{str(idx).zfill(6)}.png"), cv.IMREAD_GRAYSCALE
        )

        writer.set_idx(idx, [lung, loc, drr, bones])

    writer.set_int("length", n_items)
