from functools import partial
from glob import glob
from random import Random
from typing import Any, Dict, List, Union

import pandas as pd
from ct.meta import check_series_consistency, check_unique, parse_meta
from rich import print
from tqdm.contrib.concurrent import process_map
from utils.config import CONFIG

from lidc.bad import BAD


def parse_subject(path: str) -> int:
    return int(
        path.split("/")[-4].replace("LIDC-IDRI-", ""),
    )


def parse_id(path: str) -> str:
    return path.split("/")[-1].replace(".dcm", "")


def parse_dcm_fields():
    dcms = list(glob(f"{CONFIG.LIDC_DIR}/**/**.dcm", recursive=True))
    parse = partial(
        parse_meta,
        subject=parse_subject,
        id=parse_id,
    )
    records = process_map(parse, dcms, chunksize=1, desc="Parsing DICOM fields...")
    df = pd.DataFrame.from_records([record for record in records if record is not None])
    df.to_csv(CONFIG.OUTPUT_DIR / "lidc_meta.csv", index=False)


def select_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each subject, type pair, select series with minimum slice thickness
    """
    min_df: Any = df.iloc[df.groupby(["Subject"])["SliceThickness"].idxmin()]
    min_pairs = min_df[["Subject", "SeriesNumber"]].values.tolist()
    tmp_df = df.set_index(["Subject", "SeriesNumber"])
    return tmp_df[tmp_df.index.isin(min_pairs)].reset_index()


def select_dcms():
    # Read data and remove weird entry
    df: Any = pd.read_csv(CONFIG.OUTPUT_DIR / "lidc_meta.csv")
    df = df[~df["Subject"].isin(BAD)]
    df = df.query(
        "PatientPosition == 'FFS' & ImageType0 == 'ORIGINAL' & ImageType1 == 'PRIMARY'"
    )
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
    df = select_min(df)

    # Save info for each subject
    groups = df.groupby(["Subject"])
    paths: List[Dict[str, Union[List[str], str]]] = []
    for i, (subject, group_df) in enumerate(groups):
        sorted_df = group_df.sort_values(by="Path")
        info = {
            "overall_index": i,
            "subject": subject,
        }
        img_paths = sorted_df["Path"].tolist()
        info["lung"] = tuple(img_paths)

        paths.append(info)

    # # Shuffle paths
    Random(1035).shuffle(paths)
    print(f"Found {len(paths)} valid subjects.")

    train_df = pd.DataFrame.from_records(paths[:680])
    train_df.to_pickle(str(CONFIG.OUTPUT_DIR / "lidc_train_meta.pkl"))

    test_df = pd.DataFrame.from_records(paths[680:780])
    test_df.to_pickle(str(CONFIG.OUTPUT_DIR / "lidc_test_meta.pkl"))


if __name__ == "__main__":
    parse_dcm_fields()
    select_dcms()
