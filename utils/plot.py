from turtle import shape
from typing import Collection, Iterable, List, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact


def use_inline():
    matplotlib.use("module://matplotlib_inline.backend_inline")


def use_widget():
    matplotlib.use("ipympl")


def plot(
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
            else:
                ax = axes[j]
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if i == 0:
                ax.set_title(titles[j])
            img = img_grid[i][j]
            cmap = "gray" if len(img.shape) <= 2 else None
            ax.imshow(img, cmap=cmap)


def plot_imgs(imgs: List[np.ndarray], shape: Tuple[int, int]):
    total = np.prod(shape)
    assert total == len(imgs), "Incompatible shape"
    imgs = [
        np.concatenate(imgs[i * shape[0] : (i + 1) * shape[0]], axis=0)
        for i in range(shape[1])
    ]
    _, ax = plt.subplots(figsize=(shape[0] * 5, shape[1] * 5))
    ax.imshow(np.concatenate(imgs, axis=1), cmap="gray")


def plot_volume(
    img: np.ndarray,
    axis: int = 0,
    value_range: Tuple[int, int] = None,
    size: int = 8,
):
    assert len(img.shape) == 3, "Image is not 3D..."
    assert (
        "ipympl" in matplotlib.get_backend()
    ), "Please add %matplotlib widget to top of notebook / cell."

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
    l = plt.imshow(
        first_slice,
        vmin=value_range[0],
        vmax=value_range[1],
        cmap="gray",
    )

    # Update
    def update(val):
        l.set_data(img.take(int(val), axis=axis))
        fig.canvas.draw_idle()

    interact(update, val=IntSlider(min=0, max=img.shape[axis] - 1, value=1))


def plot_slice(
    img: np.ndarray,
    value_range: Tuple[int, int] = None,
    size: int = 8,
):
    assert len(img.shape) == 2, "Image is not 2D..."

    # Set figure pixel range
    if value_range is None:
        value_range = (img.min(), img.max())

    # Set figure size
    rows, cols = img.shape
    plt.figure(figsize=(size, int(size / cols * rows)))

    plt.imshow(img, cmap="gray")


def plot_stack(
    imgs: Iterable[np.ndarray],
    axis: int = 0,
    value_range: Tuple[int, int] = None,
    size: int = 8,
):
    img = combine(imgs, axis=axis)

    assert len(img.shape) == 2, "Image is not 2D..."

    # Set figure pixel range
    if value_range is None:
        value_range = (img.min(), img.max())

    # Set figure size
    rows, cols = img.shape
    plt.figure(figsize=(size, int(size / cols * rows)))

    plt.imshow(img, cmap="gray")
