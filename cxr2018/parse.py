import sys

sys.path.append("..")

from glob import glob
from random import Random

import cv2 as cv
from utils.config import CONFIG
from utils.cv import crop
from utils.utils import track

if __name__ == "__main__":
    (CONFIG.OUTPUT_DIR / "cxr2018/train").mkdir(parents=True, exist_ok=True)
    (CONFIG.OUTPUT_DIR / "cxr2018/test").mkdir(parents=True, exist_ok=True)

    train_imgs = list(glob(f"{CONFIG.CXR2018_DIR}/train/NORMAL/*.jpeg"))
    Random(1035).shuffle(train_imgs)
    for i, img_path in track(list(enumerate(train_imgs[:1000]))):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = crop(img, size=256)
        cv.imwrite(str(CONFIG.OUTPUT_DIR / f"cxr2018/train/{str(i).zfill(6)}.png"), img)

    test_imgs = list(glob(f"{CONFIG.CXR2018_DIR}/test/NORMAL/*.jpeg"))
    Random(1035).shuffle(test_imgs)
    for i, img_path in track(list(enumerate(test_imgs[:230]))):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = crop(img, size=256)
        cv.imwrite(str(CONFIG.OUTPUT_DIR / f"cxr2018/test/{str(i).zfill(6)}.png"), img)
