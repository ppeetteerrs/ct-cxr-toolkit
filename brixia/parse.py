import warnings
from pathlib import Path
from random import Random
from typing import Tuple

import cv2 as cv
import numpy as np
import SimpleITK as sitk
from lib.cv import crop, min_max_normalize
from tqdm.contrib.concurrent import process_map

DATA_DIR = 256
RESOLUTION = 256
OUTPUT_LMDB = True

def proc_item(arg: Tuple[int, Path, str]):
    i, dcm_path, folder = arg
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(dcm_path))
    img = reader.Execute()
    img = sitk.RescaleIntensity(img, 0, 255)
    np_img = sitk.GetArrayFromImage(img)[0].astype(np.uint8)
    np_img = min_max_normalize(crop(np_img, size=CONFIG.RESOLUTION))
    cv.imwrite(
        str(CONFIG.OUTPUT_DIR / f"brixia/{folder}/{str(i).zfill(6)}.png"), np_img
    )


def brixia_dataset():
    warnings.filterwarnings("ignore")
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    dcm_paths = list(CONFIG.BRIXIA_DIR.glob("*.dcm"))

    (CONFIG.OUTPUT_DIR / "brixia/train").mkdir(parents=True, exist_ok=True)
    (CONFIG.OUTPUT_DIR / "brixia/test").mkdir(parents=True, exist_ok=True)

    Random(1035).shuffle(dcm_paths)

    print("Generating train dataset...")
    process_map(
        proc_item,
        [(i, dcm_path, "train") for i, dcm_path in enumerate(dcm_paths[:1000])],
    )

    print("Generating test dataset...")
    process_map(
        proc_item,
        [(i, dcm_path, "test") for i, dcm_path in enumerate(dcm_paths[1000:1500])],
    )


if __name__ == "__main__":

    brixia_dataset()
