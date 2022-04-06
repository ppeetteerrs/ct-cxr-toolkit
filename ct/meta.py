from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pydicom as pyd
from rich import print


def parse_meta(
    dcm_path: str,
    subject: Union[int, Callable[[str], int]],
    id: Union[str, Callable[[str], str]],
) -> Optional[Dict[str, Any]]:
    """
    Parse DICOM metadata to dictionary
    """
    # Read Dicom data
    data = pyd.dcmread(dcm_path)

    # Get subject and image ID
    if isinstance(subject, int):
        subject_id = subject
    else:
        subject_id = subject(dcm_path)

    if isinstance(id, str):
        image_id = id
    else:
        image_id = id(dcm_path)

    if getattr(data, "SliceThickness", None) is None:
        return None

    # Return data dictionary
    return {
        "Subject": subject_id,
        "ID": image_id,
        "Path": dcm_path,
        "PhotometricInterpretation": getattr(data, "PhotometricInterpretation", np.nan),
        "SliceLocation": float(getattr(data, "SliceLocation", np.nan)),
        "SliceThickness": float(getattr(data, "SliceThickness", np.nan)),
        "ImagePositionPatient0": getattr(data, "ImagePositionPatient", [])[0],
        "ImagePositionPatient1": getattr(data, "ImagePositionPatient", [])[1],
        "ImagePositionPatient2": getattr(data, "ImagePositionPatient", [])[2],
        "PatientPosition": getattr(data, "PatientPosition", np.nan),
        "ImageOrientationPatient": getattr(data, "ImageOrientationPatient", np.nan),
        "SeriesDescription": getattr(data, "SeriesDescription", "").lower(),
        "SeriesNumber": getattr(data, "SeriesNumber", np.nan),
        "PixelSpacing0": getattr(data, "PixelSpacing", [])[0],
        "PixelSpacing1": getattr(data, "PixelSpacing", [])[1],
        "ImageType0": getattr(data, "ImageType", [])[0],
        "ImageType1": getattr(data, "ImageType", [])[1],
        "ImageType2": getattr(data, "ImageType", [])[2],
        "ImageType3": getattr(data, "ImageType", [])[3]
        if len(getattr(data, "ImageType", [])) > 3
        else "NONE",
        "HighBit": getattr(data, "HighBit", None),
    }


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
