import cv2 as cv
import pandas as pd
from utils.config import CONFIG
from utils.cv import crop, min_max_normalize
from utils.utils import track

if __name__ == "__main__":
    (CONFIG.OUTPUT_DIR / "chexpert/train").mkdir(parents=True, exist_ok=True)
    train_paths = pd.read_pickle(CONFIG.OUTPUT_DIR / "chexpert_train_meta.pkl")[
        "Path"
    ].tolist()[:1000]
    for i, train_path in track(enumerate(train_paths)):
        img = cv.imread(str(CONFIG.CHEXPERT_DIR / train_path), cv.IMREAD_GRAYSCALE)
        img = crop(min_max_normalize(img), size=CONFIG.RESOLUTION)
        cv.imwrite(
            str(CONFIG.OUTPUT_DIR / f"chexpert/train/{str(i).zfill(6)}.png"), img
        )

    (CONFIG.OUTPUT_DIR / "chexpert/test").mkdir(parents=True, exist_ok=True)
    test_paths = pd.read_pickle(CONFIG.OUTPUT_DIR / "chexpert_test_meta.pkl")[
        "Path"
    ].tolist()[:500]
    for i, test_path in track(enumerate(test_paths)):
        img = cv.imread(str(CONFIG.CHEXPERT_DIR / test_path), cv.IMREAD_GRAYSCALE)
        img = crop(min_max_normalize(img), size=CONFIG.RESOLUTION)
        cv.imwrite(str(CONFIG.OUTPUT_DIR / f"chexpert/test/{str(i).zfill(6)}.png"), img)
