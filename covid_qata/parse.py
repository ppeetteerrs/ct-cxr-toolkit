import sys

from utils.config import CONFIG

sys.path.append("..")

from glob import glob
from pathlib import Path
from random import Random

import cv2 as cv
from tqdm.rich import tqdm
from utils.cv import crop, min_max_normalize
from utils.utils import track


def covid_qata(output_dir: Path):
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    img_paths = [
        Path(item.replace("Ground-truths", "Images").replace("mask_", "")).resolve()
        for item in glob("/data/covid_qata/Ground-truths/*.png")
    ]
    print(len(img_paths))
    Random(1035).shuffle(img_paths)

    train_paths = img_paths[:1000]
    test_paths = img_paths[1000:1230]

    for idx, img_path in track(list(enumerate(train_paths))):
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        img = min_max_normalize(crop(img, size=256))
        cv.imwrite(str(output_dir / "train" / f"{str(idx).zfill(6)}.png"), img)

    for idx, img_path in track(list(enumerate(test_paths))):
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        img = min_max_normalize(crop(img, size=256))
        cv.imwrite(str(output_dir / "test" / f"{str(idx).zfill(6)}.png"), img)


if __name__ == "__main__":
    covid_qata(CONFIG.OUTPUT_DIR / "covid_qata")
