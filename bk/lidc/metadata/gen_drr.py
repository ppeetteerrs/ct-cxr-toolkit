from pathlib import Path
from typing import List, Tuple

import pandas as pd
from ct.drr import get_drr
from ct.utils import read_dcm
from utils.config import CONFIG
from utils.cv import crop, min_max_normalize, remove_border
from utils.utils import save_img, track


def lidc_drr(output_dir: Path):

    df = pd.read_pickle(output_dir / "metadata.pkl")
    paths_list: List[Tuple[str, ...]] = df["lung"].tolist()

    (output_dir / "drr").mkdir(parents=True, exist_ok=True)

    for idx, paths in track(enumerate(paths_list)):
        img = read_dcm(paths)[0]
        drr_2d = get_drr(img)
        img = crop(
            remove_border(min_max_normalize(drr_2d), tol=150), size=CONFIG.RESOLUTION
        )
        save_img(img, output_dir / f"drr/{str(idx).zfill(6)}.png")


if __name__ == "__main__":

    print("Generating train dataset...")
    lidc_drr(CONFIG.OUTPUT_DIR / "lidc/train")

    print("Generating test dataset...")
    lidc_drr(CONFIG.OUTPUT_DIR / "lidc/test")
