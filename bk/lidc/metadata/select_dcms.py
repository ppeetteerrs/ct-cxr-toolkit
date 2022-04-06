from random import Random
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from lidc.dataset.bad import BAD
from rich import print
from utils.config import CONFIG


def check_unique(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for columns with only 1 value and remove them
    """
    nunique = df.nunique(dropna=False)
    to_drop = []
    for col in df.columns:
        if nunique[col] == 1:
            print(f"Column {col} has only 1 value, dropping column...")
            to_drop.append(col)
        elif nunique[col] < 10:
            print(f"Column {col} has unique values {df[col].unique()}")
    return df.drop(to_drop, axis=1).reset_index(drop=True)


def check_series_consistency(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Check that each subject, series pair has only 1 value in selected columns, remove violating entries
    """
    nunique = df.groupby(["Subject", "SeriesNumber"])[cols].nunique(dropna=False)
    pairs: List[Tuple[str, int]] = nunique.index.values.tolist()
    bad_pairs: List[Tuple[str, int]] = []
    for subject, idx in pairs:
        pair_df = nunique.loc[(subject, idx)]
        bad_fields = pair_df[pair_df != 1].index.values.tolist()
        if len(bad_fields) > 0:
            print(
                f"Subject {subject} series {idx} has non-unique values in {bad_fields}"
            )
            bad_pairs.append((subject, idx))
    tmp_df = df.set_index(["Subject", "SeriesNumber"])
    return tmp_df[~tmp_df.index.isin(bad_pairs)].reset_index()


def select_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each subject, type pair, select series with minimum slice thickness
    """
    min_pairs = df.iloc[df.groupby(["Subject"])["SliceThickness"].idxmin()][
        ["Subject", "SeriesNumber"]
    ].values.tolist()
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

    # Separate info train and test sets
    (CONFIG.OUTPUT_DIR / "lidc/train").mkdir(parents=True, exist_ok=True)
    (CONFIG.OUTPUT_DIR / "lidc/test").mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame.from_records(paths[:680])
    train_df.to_pickle(str(CONFIG.OUTPUT_DIR / "lidc/train/metadata.pkl"))

    test_df = pd.DataFrame.from_records(paths[680:780])
    test_df.to_pickle(str(CONFIG.OUTPUT_DIR / "lidc/test/metadata.pkl"))


if __name__ == "__main__":
    select_dcms()
