"""
Run from root directory using `python -m covid_ct.generate_dataset`.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Tuple

# Make DeepDRR quiet
warnings.filterwarnings("ignore")
logging.disable()

from lib.ct.drr import get_drr
from lib.ct.localizer import get_localizer
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
    ct_paths = load("covid_ct/output/ct_paths.pkl")
    items: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = ct_paths[
        ["Subject", "med", "localizer"]
    ].values.tolist()
    seg_model = mask.get_model("unet", "R231")

    if OUTPUT_LMDB:
        (Path("covid_ct/output") / "covid_ct_lmdb").mkdir(parents=True, exist_ok=True)
        writer = LMDBImageWriter(Path("covid_ct/output") / "covid_ct_lmdb")
        writer.set_length(len(items))
    else:
        (Path("covid_ct/output") / "body").mkdir(parents=True, exist_ok=True)
        (Path("covid_ct/output") / "lung").mkdir(parents=True, exist_ok=True)
        (Path("covid_ct/output") / "bones").mkdir(parents=True, exist_ok=True)
        (Path("covid_ct/output") / "soft").mkdir(parents=True, exist_ok=True)
        (Path("covid_ct/output") / "outer").mkdir(parents=True, exist_ok=True)
        (Path("covid_ct/output") / "localizer").mkdir(parents=True, exist_ok=True)
        (Path("covid_ct/output") / "drr").mkdir(parents=True, exist_ok=True)
        writer = None

    for idx, (subject, med_paths, loc_paths) in tqdm(
        list(enumerate(items)), desc="Generating body, lung, bones, soft, outer"
    ):
        med_sitk, med_np = read_dcm(list(med_paths))

        lung_mask, body_mask, bones_mask = segment(
            med_sitk, med_np, seg_model=seg_model, lung=False
        )

        # body = project(med_np * body_mask, crop_size=RESOLUTION)
        # lung = project(med_np * lung_mask, crop_size=RESOLUTION)
        # bones = project(med_np * bones_mask, crop_size=RESOLUTION)
        soft = project(med_np * body_mask * (~bones_mask), crop_size=RESOLUTION)
        # outer = project(med_np * body_mask * (~lung_mask), crop_size=RESOLUTION)

        # Save images to different places
        if OUTPUT_LMDB:
            assert writer is not None
            # writer.set_meta((idx, "subject"), subject)
            # writer.set_img((idx, "body"), body)
            # writer.set_img((idx, "lung"), lung)
            # writer.set_img((idx, "bones"), bones)
            writer.set_img((idx, "soft"), soft)
            # writer.set_img((idx, "outer"), outer)
        else:
            # save_img(body, f"covid_ct/output/body/{str(idx).zfill(10)}.png")
            # save_img(lung, f"covid_ct/output/lung/{str(idx).zfill(10)}.png")
            # save_img(bones, f"covid_ct/output/bones/{str(idx).zfill(10)}.png")
            save_img(soft, f"covid_ct/output/soft/{str(idx).zfill(10)}.png")
            # save_img(outer, f"covid_ct/output/outer/{str(idx).zfill(10)}.png")

    # for idx, (subject, med_paths, loc_paths) in tqdm(
    #     list(enumerate(items)), desc="Generating localizer, drr"
    # ):
    #     med_sitk, med_np = read_dcm(list(med_paths))
    #     loc_np = read_dcm(loc_paths)[1][0]

    #     loc = crop(get_localizer(med_np, loc_np)[-1], size=RESOLUTION)
    #     drr = crop(remove_border(get_drr(med_sitk, move_by=750), tol=0.3), size=256)

    #     # Save images to different places
    #     if OUTPUT_LMDB:
    #         assert writer is not None
    #         # writer.set_meta((idx, "subject"), subject)
    #         writer.set_img((idx, "localizer"), loc)
    #         writer.set_img((idx, "drr"), drr)
    #     else:
    #         save_img(loc, f"covid_ct/output/localizer/{str(idx).zfill(10)}.png")
    #         save_img(drr, f"covid_ct/output/drr/{str(idx).zfill(10)}.png")
