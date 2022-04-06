from functools import partial
from glob import glob
from random import Random
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from ct.meta import check_series_consistency, check_unique, parse_meta
from rich import print
from tqdm.contrib.concurrent import process_map
from utils.config import CONFIG

from covid_ct.bad import BAD, MISALIGNED


def parse_subject(path: str) -> int:
    return int(
        path.split("/")[-2].replace("subject_", ""),
    )


def parse_id(path: str) -> str:
    return path.split("/")[-1].replace(".dcm", "")


def parse_dcm_fields():
    dcms = list(glob(f"{CONFIG.COVID_CT_DIR}/**/**.dcm"))
    parse = partial(
        parse_meta,
        subject=parse_subject,
        id=parse_id,
    )
    records = process_map(parse, dcms, chunksize=1, desc="Parsing DICOM fields...")
    df = pd.DataFrame.from_records([record for record in records if record is not None])
    df.to_csv(CONFIG.OUTPUT_DIR / "covid_ct_meta.csv", index=False)


def classify_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify images into localizer, lung and mediastinum. Filter out other types
    """
    is_localizer = df["ImageType2"] == "LOCALIZER"
    is_lung = df["SeriesDescription"].str.contains("lung", na=False)
    is_med = df["SeriesDescription"].str.contains("med", na=False)
    df["Type"] = np.where(
        is_localizer,
        "localizer",
        np.where(is_lung, "lung", np.where(is_med, "med", "others")),
    )
    df = df.query("Type != 'others'").reset_index(drop=True)
    return df


def select_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each subject, type pair, select series with minimum slice thickness
    """
    min_df: Any = df.iloc[df.groupby(["Subject", "Type"])["SliceThickness"].idxmin()]
    min_pairs = min_df[["Subject", "SeriesNumber"]].values.tolist()
    tmp_df = df.set_index(["Subject", "SeriesNumber"])
    return tmp_df[tmp_df.index.isin(min_pairs)].reset_index()


def filter_all_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select subjects with lung and med and localizer views
    """
    has_all_types = df.groupby("Subject").apply(lambda x: x["Type"].nunique() == 3)
    subjects = list(has_all_types[has_all_types].index)
    print(
        f"{len(subjects)}/{df['Subject'].nunique()} subjects has lung + med + localizer"
    )
    df: Any = df[df["Subject"].isin(subjects)]
    return df.reset_index(drop=True)


def select_dcms():
    # Read data and remove weird entry
    df = pd.read_csv(CONFIG.OUTPUT_DIR / "covid_ct_meta.csv")
    df = df.query("Subject != 133")
    df = df[(~df["Subject"].isin(BAD)) & (~df["Subject"].isin(MISALIGNED))]
    df = check_unique(df)
    df = check_series_consistency(
        df,
        [
            "ImagePositionPatient0",
            "ImagePositionPatient1",
            "ImageOrientationPatient",
            "SeriesDescription",
            "PixelSpacing0",
            "PixelSpacing1",
            "ImageType3",
        ],
    )
    df = classify_images(df)
    df = select_min(df)
    df = filter_all_types(df)

    # Save info for each subject
    groups = df.groupby(["Subject"])
    img_types = ["lung", "med", "localizer"]
    paths: List[Dict[str, Union[List[str], str]]] = []
    for i, (subject, group_df) in enumerate(groups):
        sorted_df = group_df.sort_values(by="SliceLocation")
        info = {
            "overall_index": i,
            "subject": subject,
        }
        for img_type in img_types:
            img_paths = sorted_df[sorted_df["Type"] == img_type]["Path"].tolist()
            info[img_type] = tuple(img_paths)

        paths.append(info)

    # # Shuffle paths
    Random(1035).shuffle(paths)
    print(f"Found {len(paths)} valid subjects.")

    train_df = pd.DataFrame.from_records(paths[:680])
    train_df.to_pickle(str(CONFIG.OUTPUT_DIR / "covid_ct_train_meta.pkl"))

    test_df = pd.DataFrame.from_records(paths[680:780])
    test_df.to_pickle(str(CONFIG.OUTPUT_DIR / "covid_ct_test_meta.pkl"))


if __name__ == "__main__":
    parse_dcm_fields()
    select_dcms()
