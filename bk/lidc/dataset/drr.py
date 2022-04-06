import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from ct.drr import get_drr2
from deepdrr import MobileCArm, Volume
from deepdrr.projector import Projector
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils.cv import crop, min_max_normalize
from utils.dicom import read_dcm
from utils.utils import save_img, track


def get_drr(idx: int, output_dir: Path, tmp_dir: str):

    volume = Volume.from_nifti(
        (output_dir / f"nifti/{str(idx).zfill(6)}.nii.gz"), cache_dir=Path(tmp_dir)
    )
    volume.facedown()
    carm = MobileCArm()
    carm.reposition(volume.center_in_world)
    carm.move_to(alpha=0, beta=-180, degrees=True)
    carm.move_by([0, 0, volume.center_in_world[0]])
    with Projector(volume, carm=carm) as projector:
        img = projector()
    img = (img * 255).astype(np.uint8)
    img = crop(img, size=256)
    return img


@dataclass(frozen=True)
class ProcInfo:
    idx: int
    out_dir: Path
    lung_paths: Tuple[str]


def to_nifti(info: ProcInfo):
    # Read inputs
    lung_sitk, _ = read_dcm(list(info.lung_paths))
    sitk.WriteImage(
        lung_sitk, str(info.out_dir / f"nifti/{str(info.idx).zfill(6)}.nii.gz")
    )


def lidc_drr(output_dir: Path):

    warnings.filterwarnings("ignore")
    df = pd.read_pickle(output_dir / "metadata.pkl")
    items: List[Tuple[str, ...]] = df["lung"].tolist()

    # tmp_dir = mkdtemp()
    (output_dir / "drr").mkdir(parents=True, exist_ok=True)
    (output_dir / "nifti").mkdir(parents=True, exist_ok=True)

    # infos = [ProcInfo(i, output_dir, lung_paths) for i, lung_paths in enumerate(items)]

    # process_map(to_nifti, infos)

    for i, item in tqdm(enumerate(items)):
        img = read_dcm(item)[0]
        drr_2d = get_drr2(img)
        # to_nifti(info)
        # drr_2d = get_drr(info.idx, info.out_dir, tmp_dir)
        # drr_2d = min_max_normalize(drr_2d)
        save_img(drr_2d, output_dir / f"drr/{str(i).zfill(6)}.png")
