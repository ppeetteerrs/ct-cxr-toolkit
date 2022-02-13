from os import environ as ENV
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class CONFIG:
    COVID_CT_DIR = Path(ENV["COVID_CT_DIR"])
    CHEXPERT_CT_DIR = Path(ENV["CHEXPERT_CT_DIR"])
    OUTPUT_DIR = Path(__file__).parent / "../output"


CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
