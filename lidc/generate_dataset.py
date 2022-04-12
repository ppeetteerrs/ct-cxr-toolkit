"""
Run from root directory using `python -m lidc.generate_dataset`.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Tuple

# Make DeepDRR quiet
warnings.filterwarnings("ignore")
logging.disable()

from lib.ct.drr import get_drr
from lib.ct.segmentation import segment
from lib.cv import crop, project, remove_border
from lib.dicom import read_dcm
from lib.lmdb import LMDBImageWriter
from lib.utils import load, save_img
from lungmask import mask
from tqdm import tqdm

RESOLUTION = 256
OUTPUT_LMDB = True

if __name__ == "__main__":
    ct_paths = load("lidc/output/ct_paths.pkl")
    items: List[Tuple[str, Tuple[str, ...]]] = ct_paths[
        ["Subject", "lung"]
    ].values.tolist()
    seg_model = mask.get_model("unet", "R231")

    if OUTPUT_LMDB:
        (Path("lidc/output") / "lmdb").mkdir(parents=True, exist_ok=True)
        writer = LMDBImageWriter(Path("lidc/output") / "lmdb")
        writer.set_length(len(items))
    else:
        (Path("lidc/output") / "body").mkdir(parents=True, exist_ok=True)
        (Path("lidc/output") / "lung").mkdir(parents=True, exist_ok=True)
        (Path("lidc/output") / "bones").mkdir(parents=True, exist_ok=True)
        (Path("lidc/output") / "outer").mkdir(parents=True, exist_ok=True)
        (Path("lidc/output") / "drr").mkdir(parents=True, exist_ok=True)
        writer = None

    for idx, (subject, lung_paths) in tqdm(
        list(enumerate(items)), desc="Generating body, lung, bones, outer"
    ):
        lung_sitk, lung_np = read_dcm(list(lung_paths), clip=True)

        lung_mask, body_mask, bones_mask = segment(
            lung_sitk, lung_np, seg_model=seg_model
        )

        body = project(lung_np * body_mask, crop_size=RESOLUTION)
        lung = project(lung_np * lung_mask, crop_size=RESOLUTION)
        bones = project(lung_np * bones_mask, crop_size=RESOLUTION)
        outer = project(lung_np * body_mask * (~lung_mask), crop_size=RESOLUTION)

        # Save images to different places
        if OUTPUT_LMDB:
            assert writer is not None
            writer.set_meta((idx, "subject"), subject)
            writer.set_img((idx, "body"), body)
            writer.set_img((idx, "lung"), lung)
            writer.set_img((idx, "bones"), bones)
            writer.set_img((idx, "outer"), outer)
        else:
            save_img(body, f"lidc/output/body/{str(idx).zfill(10)}.png")
            save_img(lung, f"lidc/output/lung/{str(idx).zfill(10)}.png")
            save_img(bones, f"lidc/output/bones/{str(idx).zfill(10)}.png")
            save_img(outer, f"lidc/output/outer/{str(idx).zfill(10)}.png")

    for idx, (subject, lung_paths) in tqdm(
        list(enumerate(items)), desc="Generating localizer, drr"
    ):
        lung_sitk, lung_np = read_dcm(list(lung_paths), clip=True)
        drr = crop(remove_border(get_drr(lung_sitk, move_by=750), tol=0.3), size=256)

        # Save images to different places
        if OUTPUT_LMDB:
            assert writer is not None
            # writer.set_meta((idx, "subject"), subject)
            writer.set_img((idx, "drr"), drr)
        else:
            save_img(drr, f"lidc/output/drr/{str(idx).zfill(10)}.png")
