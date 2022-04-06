from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from covid_ct.dataset.body_mask import get_body_mask
from covid_ct.dataset.bone_mask import get_bone_mask
from covid_ct.dataset.localizer import align_features, crop_localizer, get_feature
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


def process_slice(lung_2d: np.ndarray):
    # Extract patient body from bed
    body_2db, body_thres_2db, body_contour_2db = get_body_mask(
        lung_2d, body_threshold=35
    )
    body_2d = np.where(body_2db, lung_2d, 0).astype(np.uint8)

    # Extract bone mask
    bones_2db, bones_thres_2db, bones_contour_2db = get_bone_mask(body_2d, to_8bit(250))
    return (
        body_2d,
        bones_2db,
        body_thres_2db,
        body_contour_2db,
        bones_thres_2db,
        bones_contour_2db,
    )


def process_lung(lung_3d: np.ndarray, loc_2d: np.ndarray):
    # Get localizer
    lung_frontal_2d = project(lung_3d)
    lung_feature_2db = get_feature(lung_frontal_2d, percentage=40)
    loc_feature_2db = get_feature(loc_2d, percentage=20)

    feature_coords = align_features(lung_feature_2db, loc_feature_2db)
    loc_cropped_2d, loc_box_2d = crop_localizer(loc_2d, feature_coords)

    (
        body_3d,
        bones_3db,
        body_thres_3db,
        body_contour_3db,
        bones_thres_3db,
        bones_contour_3db,
    ) = zip(*[process_slice(lung_2d) for lung_2d in lung_3d])

    raw_2d = crop(lung_frontal_2d, size=256)
    loc_cropped_2d = min_max_normalize(crop(loc_cropped_2d, size=256))
    body_2d = project(body_3d, size=256)
    bones_2d = project(bones_3db, size=256, equalized=True)

    return raw_2d, loc_cropped_2d, body_2d, bones_2d


def covid_ct_dataset(output_dir: Path):
    df = pd.read_pickle(output_dir / "metadata.pkl")
    items: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = df[
        ["lung", "localizer"]
    ].values.tolist()

    (output_dir / "localizer").mkdir(parents=True, exist_ok=True)
    (output_dir / "lung").mkdir(parents=True, exist_ok=True)
    (output_dir / "bones").mkdir(parents=True, exist_ok=True)
    (output_dir / "drr").mkdir(parents=True, exist_ok=True)
    (output_dir / "nifti").mkdir(parents=True, exist_ok=True)

    samples: List[np.ndarray] = []

    for idx, (lung_paths, loc_paths) in track(
        list(enumerate(items)), description="Processing"
    ):
        _, lung_3d = read_dcm(list(lung_paths))
        lung_3d = normalize_hu(lung_3d)
        loc_2d = normalize_hu(read_dcm(list(loc_paths))[1][0])

        # Get lung, localizer, body, bones, drr
        raw_2d, loc_cropped_2d, body_2d, bones_2d = process_lung(lung_3d, loc_2d)

        # Save images to different places
        save_img(loc_cropped_2d, output_dir / f"localizer/{str(idx).zfill(6)}.png")
        save_img(body_2d, output_dir / f"lung/{str(idx).zfill(6)}.png")
        save_img(bones_2d, output_dir / f"bones/{str(idx).zfill(6)}.png")

        samples.append(concat((raw_2d, loc_cropped_2d, body_2d, bones_2d), axis=1))

    save_img(
        concat(
            samples,
            axis=0,
        ),
        output_dir / "samples.png",
    )
