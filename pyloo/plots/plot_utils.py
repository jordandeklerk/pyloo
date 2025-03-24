"""Utilities for plotting based on Arviz."""

import importlib
import warnings
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.colors import to_hex

from ..rcparams import rcParams

KwargSpec = dict[str, Any]


def _scale_fig_size(figsize, textsize, rows=1, cols=1):
    """Scale figure properties according to rows and cols.

    Parameters
    ----------
    figsize : float or None
        Size of figure in inches
    textsize : float or None
        fontsize
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    figsize : float or None
        Size of figure in inches
    ax_labelsize : int
        fontsize for axes label
    titlesize : int
        fontsize for title
    xt_labelsize : int
        fontsize for axes ticks
    linewidth : int
        linewidth
    markersize : int
        markersize
    """
    params = mpl.rcParams
    rc_width, rc_height = tuple(params["figure.figsize"])
    rc_ax_labelsize = params["axes.labelsize"]
    rc_titlesize = params["axes.titlesize"]
    rc_xt_labelsize = params["xtick.labelsize"]
    rc_linewidth = params["lines.linewidth"]
    rc_markersize = params["lines.markersize"]
    if isinstance(rc_ax_labelsize, str):
        rc_ax_labelsize = 15
    if isinstance(rc_titlesize, str):
        rc_titlesize = 16
    if isinstance(rc_xt_labelsize, str):
        rc_xt_labelsize = 14

    if figsize is None:
        width, height = rc_width, rc_height
        sff = 1 if (rows == cols == 1) else 1.15
        width = width * cols * sff
        height = height * rows * sff
    else:
        width, height = figsize

    if textsize is not None:
        scale_factor = textsize / rc_xt_labelsize
    elif rows == cols == 1:
        scale_factor = ((width * height) / (rc_width * rc_height)) ** 0.5
    else:
        scale_factor = 1

    ax_labelsize = rc_ax_labelsize * scale_factor
    titlesize = rc_titlesize * scale_factor
    xt_labelsize = rc_xt_labelsize * scale_factor
    linewidth = rc_linewidth * scale_factor
    markersize = rc_markersize * scale_factor

    return (width, height), ax_labelsize, titlesize, xt_labelsize, linewidth, markersize


def default_grid(n_items, grid=None, max_cols=4, min_cols=3):
    """Make a grid for subplots.

    Tries to get as close to sqrt(n_items) x sqrt(n_items) as it can,
    but allows for custom logic

    Parameters
    ----------
    n_items : int
        Number of panels required
    grid : tuple
        Number of rows and columns
    max_cols : int
        Maximum number of columns, inclusive
    min_cols : int
        Minimum number of columns, inclusive

    Returns
    -------
    (int, int)
        Rows and columns, so that rows * columns >= n_items
    """
    if grid is None:

        def in_bounds(val):
            return np.clip(val, min_cols, max_cols)

        if n_items <= max_cols:
            return 1, n_items
        ideal = in_bounds(round(n_items**0.5))

        for offset in (0, 1, -1, 2, -2):
            cols = in_bounds(ideal + offset)
            rows, extra = divmod(n_items, cols)
            if extra == 0:
                return rows, cols
        return n_items // ideal + 1, ideal
    else:
        rows, cols = grid
        if rows * cols < n_items:
            raise ValueError(
                "The number of rows times columns is less than the number of subplots"
            )
        if (rows * cols) - n_items >= cols:
            warnings.warn(
                "The number of rows times columns is larger than necessary",
                UserWarning,
                stacklevel=2,
            )
        return rows, cols


def get_plotting_function(plot_name, plot_module, backend):
    """Return plotting function for correct backend.

    Parameters
    ----------
    plot_name : str
        Name of the plotting function
    plot_module : str
        Name of the module where the plotting function is defined
    backend : str
        Name of the backend to use

    Returns
    -------
    callable
        The plotting function
    """
    _backend = {
        "mpl": "matplotlib",
        "bokeh": "bokeh",
        "matplotlib": "matplotlib",
    }

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    try:
        backend = _backend[backend]
    except KeyError as err:
        raise KeyError(
            f"Backend {backend} is not implemented. Try backend in"
            f" {set(_backend.values())}"
        ) from err

    # Perform import of plotting method
    module = importlib.import_module(f"pyloo.plots.backends.{backend}.{plot_module}")
    plotting_method = getattr(module, plot_name)

    return plotting_method


def format_sig_figs(value, default=None):
    """Get a default number of significant figures.

    Gives the integer part or `default`, whichever is bigger.

    Examples
    --------
    0.1234 --> 0.12
    1.234  --> 1.2
    12.34  --> 12
    123.4  --> 123
    """
    if default is None:
        default = 2
    if value == 0:
        return 1
    return max(int(np.log10(np.abs(value))) + 1, default)


def round_num(n, round_to):
    """Return a string representing a number with `round_to` significant figures.

    Parameters
    ----------
    n : float
        number to round
    round_to : int
        number of significant figures
    """
    sig_figs = format_sig_figs(n, round_to)
    return "{n:.{sig_figs}g}".format(n=n, sig_figs=sig_figs)


def vectorized_to_hex(c_values, keep_alpha=False):
    """Convert a color (including vector of colors) to hex.

    Parameters
    ----------
    c: Matplotlib color

    keep_alpha: boolean
        to select if alpha values should be kept in the final hex values.

    Returns
    -------
    rgba_hex : vector of hex values
    """
    try:
        hex_color = to_hex(c_values, keep_alpha)
    except ValueError:
        hex_color = [to_hex(color, keep_alpha) for color in c_values]
    return hex_color


def _init_kwargs_dict(kwargs):
    """Initialize kwargs dict.

    If the input is a dictionary, it returns
    a copy of the dictionary, otherwise it
    returns an empty dictionary.

    Parameters
    ----------
    kwargs : dict or None
        kwargs dict to initialize
    """
    return {} if kwargs is None else kwargs.copy()
