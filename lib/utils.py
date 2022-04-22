import pickle
from pathlib import Path
from typing import Any, TypeVar, Union

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

ArrB = NDArray[np.bool8]
Arr8U = NDArray[np.uint8]
Arr16U = NDArray[np.uint16]
Arr32F = NDArray[np.float32]
AnyArr = TypeVar("AnyArr", bound=np.ndarray)


def load(path: Union[str, Path]) -> Any:
    return pickle.load(open(path, "rb"))


def save_img(img: Arr32F, path: Union[str, Path]):
    cv.imwrite(str(path), (img * 255).astype(np.uint8))


def nop(item: Any, *_: Any, **__: Any) -> Any:
    return item
