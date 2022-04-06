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

        elapsed_text = total_text = "---"

        if elapsed is not None:
            elapsed_text = timedelta(seconds=int(elapsed + 0.5))

            if remaining is not None:
                total_text = timedelta(seconds=int(elapsed + remaining + 0.5))

        return Text(f"[{elapsed_text}/{total_text}]", style="progress.remaining")


class StepProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:

        return Text(
            f"{int(task.completed)}/{int(task.total)}", style="progress.percentage"
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

    sequence = list(sequence)
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
