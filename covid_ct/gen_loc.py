from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from ct.drr import get_drr
from ct.localizer import align_features, crop_localizer, get_feature
from ct.masks import get_body_mask, get_bone_mask
from ct.utils import normalize_hu, project, read_dcm, to_8bit
from utils.config import CONFIG
from utils.cv import concat, crop, min_max_normalize, remove_border
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


def process_lung(lung_3d: np.ndarray, loc_2d: np.ndarray):
    # Get localizer
    lung_frontal_2d = project(lung_3d)
    lung_feature_2db = get_feature(lung_frontal_2d, percentage=40)
    loc_feature_2db = get_feature(loc_2d, percentage=20)

    feature_coords = align_features(lung_feature_2db, loc_feature_2db)
    loc_cropped_2d, loc_box_2d = crop_localizer(loc_2d, feature_coords)

    raw_2d = min_max_normalize(crop(lung_frontal_2d, size=CONFIG.RESOLUTION))
    loc_cropped_2d = min_max_normalize(crop(loc_cropped_2d, size=CONFIG.RESOLUTION))
    return raw_2d, loc_cropped_2d


def process_lung_all(lung_3d: np.ndarray, loc_2d: np.ndarray):
    # Get localizer
    lung_frontal_2d = project(lung_3d)
    lung_feature_2db = get_feature(lung_frontal_2d, percentage=40)
    loc_feature_2db = get_feature(loc_2d, percentage=20)

    feature_coords = align_features(lung_feature_2db, loc_feature_2db)
    loc_cropped_2d, loc_box_2d = crop_localizer(loc_2d, feature_coords)

    raw_2d = min_max_normalize(crop(lung_frontal_2d, size=CONFIG.RESOLUTION))
    loc_cropped_2d = min_max_normalize(crop(loc_cropped_2d, size=CONFIG.RESOLUTION))
    return raw_2d, lung_feature_2db, loc_feature_2db, loc_box_2d, loc_cropped_2d


def covid_ct_dataset(df_path: Path, output_dir: Path):
    df = pd.read_pickle(df_path)
    items: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = df[
        ["lung", "localizer"]
    ].values.tolist()

    (output_dir / "localizer").mkdir(parents=True, exist_ok=True)

    for idx, (lung_paths, loc_paths) in track(
        enumerate(items), description="Processing"
    ):
        lung_sitk, lung_3d = read_dcm(list(lung_paths))

        # Get lung, localizer, body, bones, soft
        lung_3d = normalize_hu(lung_3d)
        loc_2d = normalize_hu(read_dcm(list(loc_paths))[1][0])

        # Get lung, localizer, body, bones, drr
        raw_2d, loc_cropped_2d = process_lung(lung_3d, loc_2d)

        # Save images to different places
        save_img(
            concat([raw_2d, loc_cropped_2d]),
            output_dir / f"localizer/{str(idx).zfill(6)}.png",
        )


if __name__ == "__main__":

    print("Generating train dataset...")
    covid_ct_dataset(
        CONFIG.OUTPUT_DIR / "covid_ct_loc.pkl",
        CONFIG.OUTPUT_DIR / "covid_ct/train",
    )

    # print("Generating test dataset...")
    # covid_ct_dataset(
    #     CONFIG.OUTPUT_DIR / "covid_ct_test_meta.pkl",
    #     CONFIG.OUTPUT_DIR / "covid_ct/test",
    # )
