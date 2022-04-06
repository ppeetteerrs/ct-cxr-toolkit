from os import environ as ENV
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class CONFIG:
    RESOLUTION = int(ENV["RESOLUTION"])
    COVID_CT_DIR = Path(ENV["COVID_CT_DIR"])
    CHEXPERT_DIR = Path(ENV["CHEXPERT_DIR"])
    LIDC_DIR = Path(ENV["LIDC_DIR"])
    CXR2018_DIR = Path(ENV["CXR2018_DIR"])
    BRIXIA_DIR = Path(ENV["BRIXIA_DIR"])
    OUTPUT_DIR = Path(__file__).parent / "../output"


CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
