from glob import glob
from typing import Any, Dict, Optional

import pandas as pd
import pydicom as pyd
from tqdm.contrib.concurrent import process_map
from utils.config import CONFIG


def parse_record(dcm_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse DICOM metadata to dictionary
    """
    # Read Dicom data
    data = pyd.dcmread(dcm_path)

    # Get subject and image ID
    subject_id = dcm_path.split("/")[-4].replace("LIDC-IDRI-", "")
    image_id = dcm_path.split("/")[-1].replace(".dcm", "")

    if getattr(data, "SliceThickness", None) is None:
        return None

    # Return data dictionary
    return {
        "Subject": subject_id,
        "ID": image_id,
        "Path": dcm_path,
        "PhotometricInterpretation": data.PhotometricInterpretation,
        "SliceLocation": float(data.SliceLocation),
        "SliceThickness": float(data.SliceThickness),
        "ImagePositionPatient0": data.ImagePositionPatient[0],
        "ImagePositionPatient1": data.ImagePositionPatient[1],
        "ImagePositionPatient2": data.ImagePositionPatient[2],
        "PatientPosition": data.PatientPosition,
        "ImageOrientationPatient": tuple(data.ImageOrientationPatient),
        "SeriesDescription": getattr(data, "SeriesDescription", "").lower(),
        "SeriesNumber": data.SeriesNumber,
        "PixelSpacing0": data.PixelSpacing[0],
        "PixelSpacing1": data.PixelSpacing[1],
        "ImageType0": data.ImageType[0],
        "ImageType1": data.ImageType[1],
        "ImageType2": data.ImageType[2],
        "ImageType3": data.ImageType[3] if len(data.ImageType) > 3 else "NONE",
        "HighBit": data.HighBit,
    }


def parse_dcm_fields():
    dcms = list(glob(f"{CONFIG.LIDC_DIR}/**/**.dcm", recursive=True))
    records = process_map(parse_record, dcms, chunksize=1)
    df = pd.DataFrame.from_records([item for item in records if item is not None])
    df.to_csv(CONFIG.OUTPUT_DIR / "lidc_meta.csv", index=False)


if __name__ == "__main__":
    parse_dcm_fields()
