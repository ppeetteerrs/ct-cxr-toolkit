import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, TypeVar, Union

import cv2 as cv
import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.text import Text

from utils.config import CONFIG


def load(path: Union[str, Path]) -> Any:
    return pickle.load(open(path, "rb"))


def load_df(path: str) -> pd.DataFrame:
    return pd.read_pickle(CONFIG.OUTPUT_DIR / path)


def save_img(img: np.ndarray, path: Path):
    cv.imwrite(str(path), img)


T = TypeVar("T")


class TimeProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining

        elapsed_text = total_text = "-:--:--.---"

        if elapsed is not None:
            elapsed_text = str(timedelta(milliseconds=int(elapsed * 1000)))[:-3]

            if remaining is not None:
                total_text = str(
                    timedelta(milliseconds=int(elapsed * 1000) + int(remaining * 1000))
                )[:-3]

        return Text(f"[{elapsed_text} / {total_text}]", style="progress.remaining")


class StepProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:

        return Text(
            f"{int(task.completed)} / {int(task.total)}", style="progress.percentage"
        )


class SpeedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed

        if elapsed is None or elapsed == 0:
            return Text("", style="progress.elapsed")

        it_per_sec = task.completed / elapsed

        if it_per_sec > 1:
            return Text(f"{it_per_sec:3.3f}it/s", style="progress.elapsed")
        elif it_per_sec > 0:
            return Text(f"{1/it_per_sec:3.3f}s/it", style="progress.elapsed")
        else:
            return Text("", style="progress.elapsed")


def track(
    sequence: Union[Sequence[T], Iterable[T]],
    description: str = "Working...",
    total: Optional[float] = None,
    transient: bool = False,
) -> Iterable[T]:

    if isinstance(sequence, Sequence):
        total = total or len(sequence)

    columns: List["ProgressColumn"] = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(
            bar_width=None,
            style="bar.back",
            complete_style="bar.complete",
            finished_style="bar.finished",
            pulse_style="bar.pulse",
        ),
        StepProgressColumn(),
        TimeProgressColumn(),
        SpeedColumn(),
    ]
    progress = Progress(
        *columns,
        auto_refresh=True,
        console=None,
        transient=transient,
        get_time=None,
        refresh_per_second=10,
        disable=False,
    )

    with progress:
        yield from progress.track(
            sequence,
            total=total,
            description=description,
            update_period=0.1,
        )


COVID_CT_BAD = [
    5,
    6,
    7,
    9,
    11,
    25,
    35,
    73,
    76,
    80,
    81,
    82,
    88,
    90,
    108,
    109,
    113,
    114,
    124,
    126,
    127,
    128,
    129,
    130,
    136,
    139,
    145,
    156,
    158,
    160,
    177,
    180,
    191,
    193,
    199,
    212,
    215,
    217,
    240,
    241,
    243,
    244,
    257,
    270,
    277,
    282,
    290,
    323,
    341,
    357,
    362,
    397,
    421,
    432,
    500,
    515,
    516,
    517,
    518,
    536,
    572,
    580,
    585,
    604,
    605,
    606,
    613,
    650,
    652,
    654,
    659,
    667,
    668,
    669,
    670,
    671,
    685,
    743,
    748,
    782,
    790,
    834,
    835,
    852,
    853,
    870,
    988,
    1009,
    1012,
    1036,
    1039,
    1041,
    1049,
    1051,
    1052,
    1053,
    1073,
    1085,
    1089,
    1106,
]

COVID_CT_MISALIGNED = [
    3,
    15,
    92,
    115,
    116,
    120,
    121,
    137,
    154,
    155,
    169,
    175,
    195,
    211,
    214,
    218,
    231,
    256,
    260,
    364,
    371,
    381,
    388,
    394,
    403,
    414,
    418,
    423,
    437,
    439,
    441,
    442,
    445,
    454,
    468,
    471,
    475,
    509,
    512,
    537,
    558,
    563,
    564,
    568,
    574,
    615,
    679,
    701,
    702,
    708,
    715,
    753,
    757,
    844,
    851,
    1028,
    1062,
    1098,
    1099,
    1102,
]
