import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from deepdrr import MobileCArm, Volume
from deepdrr.projector import Projector
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils.cv import crop, min_max_normalize
from utils.dicom import read_dcm
from utils.utils import save_img, track

warnings.filterwarnings("ignore")


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


def covid_ct_drr(output_dir: Path):
    df = pd.read_pickle(output_dir / "metadata.pkl")
    items: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = df[
        ["lung", "localizer"]
    ].values.tolist()

    tmp_dir = mkdtemp()
    (output_dir / "localizer").mkdir(parents=True, exist_ok=True)
    (output_dir / "lung").mkdir(parents=True, exist_ok=True)
    (output_dir / "bones").mkdir(parents=True, exist_ok=True)
    (output_dir / "drr").mkdir(parents=True, exist_ok=True)
    (output_dir / "nifti").mkdir(parents=True, exist_ok=True)

    infos = [
        ProcInfo(i, output_dir, lung_paths) for i, (lung_paths, _) in enumerate(items)
    ]

    process_map(to_nifti, infos)

    for info in tqdm(infos):
        drr_2d = get_drr(info.idx, info.out_dir, tmp_dir)
        drr_2d = min_max_normalize(drr_2d)
        save_img(drr_2d, info.out_dir / f"drr/{str(info.idx).zfill(6)}.png")
