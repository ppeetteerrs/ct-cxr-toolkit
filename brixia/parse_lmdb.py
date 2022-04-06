import warnings
from multiprocessing import Pool
from pathlib import Path
from random import Random
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from utils.config import CONFIG
from utils.cv import crop, min_max_normalize
from utils.lmdb import LMDBImageWriter, chexpert_indexer


def proc_item(arg: Tuple[int, Path]) -> Tuple[int, np.ndarray]:
    idx, dcm_path = arg
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(dcm_path))
    img = reader.Execute()
    img = sitk.RescaleIntensity(img, 0, 255)
    np_img = sitk.GetArrayFromImage(img)[0].astype(np.uint8)
    np_img = min_max_normalize(crop(np_img, size=CONFIG.RESOLUTION))
    return idx, np_img


def brixia_dataset():
    warnings.filterwarnings("ignore")
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    dcm_paths = list(CONFIG.BRIXIA_DIR.glob("*.dcm"))

    (CONFIG.OUTPUT_DIR / "brixia/lmdb").mkdir(parents=True, exist_ok=True)

    Random(1035).shuffle(dcm_paths)

    writer = LMDBImageWriter(CONFIG.OUTPUT_DIR / "brixia/lmdb", chexpert_indexer)

    with Pool(6) as pool:
        for idx, val in tqdm(
            pool.imap_unordered(proc_item, enumerate(dcm_paths)),
            total=len(dcm_paths),
            dynamic_ncols=True,
            smoothing=0.01,
        ):
            writer.set_idx(idx, [val])

    writer.set_int("length", len(dcm_paths))


if __name__ == "__main__":

    brixia_dataset()
