from typing import List, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact


def use_inline():
    matplotlib.use("module://matplotlib_inline.backend_inline")


def use_widget():
    matplotlib.use("module://ipympl.backend_nbagg")


def plot_grid(
    imgs: Union[List[np.ndarray], List[List[np.ndarray]]], titles: List[str] = None
):
    use_inline()

    img_grid: List[List[np.ndarray]]

    if not isinstance(imgs[0], List):
        img_grid = cast(List[List[np.ndarray]], [imgs])
    else:
        img_grid = cast(List[List[np.ndarray]], imgs)

    n_rows = len(img_grid)
    n_cols = len(img_grid[0])

    plot_w = max(6, 5 * n_cols)
    plot_h = max(6, 5 * n_rows)

    titles = ["" for _ in range(n_cols)] if titles is None else titles
    _, axes = plt.subplots(n_rows, n_cols, figsize=(plot_w, plot_h))

    for i in range(n_rows):
        for j in range(n_cols):
            if n_rows > 1:
                ax = axes[i][j]
            elif n_cols > 1:
                ax = axes[j]
            else:
                ax = axes
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if i == 0:
                ax.set_title(titles[j])
            img = img_grid[i][j]
            cmap = "gray" if len(img.shape) <= 2 else None
            ax.imshow(img, cmap=cmap)


def plot_volume(
    img: np.ndarray,
    axis: int = 0,
    value_range: Tuple[int, int] = None,
    size: int = 8,
):
    use_widget()

    if isinstance(img, List):
        img = np.array(img)

    # Set figure pixel range
    if value_range is None:
        value_range = (img.min(), img.max())

    # Set figure size
    first_slice = img.take(0, axis=axis)
    rows, cols = first_slice.shape
    fig = plt.figure(figsize=(size, int(size / cols * rows)))

    # Set figure rubbish to False
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    # Show first slice
    ax = plt.imshow(
        first_slice,
        vmin=value_range[0],
        vmax=value_range[1],
        cmap="gray",
    )

    # Update
    def update(val):
        ax.set_data(img.take(int(val), axis=axis))
        fig.canvas.draw_idle()

    interact(update, val=IntSlider(min=0, max=img.shape[axis] - 1, value=1))
