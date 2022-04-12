"""
Run from root directory using `python -m chexpert.generate_dataset`.
"""

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import pandas as pd
from lib.lmdb import LMDBImageWriter
from tqdm import tqdm

from chexpert.utils import CheXpertImg

CHEXPERT_DIR = Path("/data/chexpert")
RESOLUTION = 256


def proc_img(
    img_dir: Path,
    img: CheXpertImg,
) -> Tuple[CheXpertImg, np.ndarray]:
    return img, img.proc_img(RESOLUTION, img_dir)


def proc_df(df: pd.DataFrame, img_dir: Path, lmdb_dir: Path):
    # Turn dataframe rows into metadata object
    imgs = [
        CheXpertImg(idx, path, sex, age, others)
        for idx, path, sex, age, *others in tqdm(
            df.itertuples(), dynamic_ncols=True, smoothing=0.01
        )
    ]
    print(f"Got {len(imgs)} images.")

    writer = LMDBImageWriter(lmdb_dir)

    with Pool(4) as pool:
        for img, val in tqdm(
            pool.imap_unordered(partial(proc_img, img_dir), imgs),
            total=len(imgs),
            dynamic_ncols=True,
            smoothing=0.01,
        ):
            writer.set_img(img.idx, val)

    writer.set_length(len(imgs))


if __name__ == "__main__":
    # Read train.csv and valid.csv
    train_df = cast(pd.DataFrame, pd.read_csv(CHEXPERT_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(CHEXPERT_DIR / "valid.csv"))
    df = pd.concat([train_df, test_df])

    # Remove root path of extracted folder
    df["Path"] = df["Path"].str.replace("CheXpert-v1.0/", "", regex=False)
    df["Support Devices"] = df["Support Devices"].fillna(-1)

    # Select Frontal AP CXR without Support Devices
    interested_df = df[
        (df["Frontal/Lateral"] == "Frontal")
        & (df["AP/PA"] == "AP")
        & (df["Support Devices"] <= 0)
    ]
    interested_df = interested_df.drop(
        ["Frontal/Lateral", "AP/PA", "No Finding"], axis=1
    )

    # Drop patients without path / sex / age and fill in missing pathlogies as -1
    interested_df = (
        interested_df.dropna(subset=["Path", "Sex", "Age"])
        .fillna(-1)
        .reset_index(drop=True)
    )
    print(f"Total of {interested_df.shape[0]} images left")

    Path("chexpert/output").mkdir(parents=True, exist_ok=True)
    pd.to_pickle(interested_df, str("chexpert/output/metadata.pkl"))

    proc_df(interested_df, CHEXPERT_DIR, Path("chexpert/output/lmdb"))
