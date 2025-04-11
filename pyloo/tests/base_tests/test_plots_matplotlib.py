"""Tests for matplotlib plotting functionality."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ... import loo
from ...plots import plot_influence, plot_loo, plot_loo_difference


@pytest.fixture(scope="function", autouse=True)
def clean_plots(request, save_dir):
    """Close plots after each test, optionally save if --save is specified during test invocation."""

    def fin():
        if save_dir is not None:
            plt.savefig(f"{os.path.join(save_dir, request.node.name)}.png")
        plt.close("all")

    request.addfinalizer(fin)


@pytest.fixture(scope="module")
def loo_data(centered_eight):
    """Create test data for LOO plots."""
    loo_results = loo(centered_eight, pointwise=True)
    return loo_results


@pytest.fixture(scope="module")
def loo_data_pair(centered_eight, non_centered_eight):
    """Create a pair of LOO results for comparison plots."""
    loo_result1 = loo(centered_eight, pointwise=True)
    loo_result2 = loo(non_centered_eight, pointwise=True)

    return loo_result1, loo_result2


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_name": None},
        {"color": "red"},
        {"threshold": 0.5},
        {"sort": False},
        {"n_points": 5},
        {"n_points": -5},
        {"use_pareto_k": False},
        {"k_threshold": 0.5},
        {"figsize": (8, 6)},
        {"textsize": 12},
    ],
)
def test_plot_influence(loo_data, kwargs):
    """Test influence plot with various parameters."""
    ax = plot_influence(loo_data, **kwargs)
    assert ax is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_name": None},
        {"color": "blue"},
        {"threshold": 0.7},
        {"show_pareto_k": False},
        {"show_elpd": True},
        {"figsize": (8, 6)},
        {"textsize": 12},
    ],
)
def test_plot_loo(loo_data, kwargs):
    """Test LOO plot with various parameters."""
    ax = plot_loo(loo_data, **kwargs)
    assert ax is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"group": None},
        {"outlier_thresh": 2.0},
        {"size": 2},
        {"alpha": 0.7},
        {"jitter": 0.1},
        {"jitter": (0.1, 0.2)},
        {"sort_by_group": True, "group": np.array(["Group A", "Group B"] * 4)},
        {"figsize": (8, 6)},
        {"textsize": 12},
    ],
)
def test_plot_loo_difference(loo_data_pair, kwargs):
    """Test LOO difference plot with various parameters."""
    loo_result1, loo_result2 = loo_data_pair

    x_values = np.arange(len(loo_result1.loo_i))

    ax = plot_loo_difference(x_values, loo_result1, loo_result2, **kwargs)
    assert ax is not None


def test_plot_loo_difference_validation(loo_data_pair):
    """Test LOO difference plot input validation."""
    loo_result1, loo_result2 = loo_data_pair

    x_values_wrong = np.arange(len(loo_result1.loo_i) - 1)

    with pytest.raises(ValueError, match="Length of x_values"):
        plot_loo_difference(x_values_wrong, loo_result1, loo_result2)

    group_wrong = ["A"] * (len(loo_result1.loo_i) - 1)

    with pytest.raises(ValueError, match="Length of group"):
        x_values = np.arange(len(loo_result1.loo_i))
        plot_loo_difference(x_values, loo_result1, loo_result2, group=group_wrong)


def test_plot_backend_kwarg(loo_data):
    """Test backend_kwargs are properly passed to the plotting functions."""
    backend_kwargs = {"dpi": 100}

    ax = plot_influence(loo_data, backend_kwargs=backend_kwargs)
    assert ax is not None

    ax = plot_loo(loo_data, backend_kwargs=backend_kwargs)
    assert ax is not None


def test_plots_ax_argument(loo_data, loo_data_pair):
    """Test passing axis to plotting functions."""
    _, ax_influence = plt.subplots()
    _, ax_loo = plt.subplots()
    _, ax_loo_diff = plt.subplots()

    result_ax = plot_influence(loo_data, ax=ax_influence)
    assert result_ax is ax_influence

    result_ax = plot_loo(loo_data, ax=ax_loo)
    assert result_ax is ax_loo

    loo_result1, loo_result2 = loo_data_pair
    x_values = np.arange(len(loo_result1.loo_i))
    result_ax = plot_loo_difference(x_values, loo_result1, loo_result2, ax=ax_loo_diff)
    assert result_ax is ax_loo_diff


def test_plot_influence_show_parameter(loo_data):
    """Test show parameter for plot_influence."""
    ax = plot_influence(loo_data, show=False)
    assert ax is not None


def test_plot_loo_show_parameter(loo_data):
    """Test show parameter for plot_loo."""
    ax = plot_loo(loo_data, show=False)
    assert ax is not None


def test_plot_loo_difference_show_parameter(loo_data_pair):
    """Test show parameter for plot_loo_difference."""
    loo_result1, loo_result2 = loo_data_pair
    x_values = np.arange(len(loo_result1.loo_i))
    ax = plot_loo_difference(x_values, loo_result1, loo_result2, show=False)
    assert ax is not None


def test_plot_influence_style_parameter(loo_data):
    """Test style parameter for plot_influence."""
    ax = plot_influence(loo_data, style="default")
    assert ax is not None

    ax = plot_influence(loo_data, style=None)
    assert ax is not None


def test_plot_loo_style_parameter(loo_data):
    """Test style parameter for plot_loo."""
    ax = plot_loo(loo_data, style="default")
    assert ax is not None

    ax = plot_loo(loo_data, style=None)
    assert ax is not None


def test_plot_loo_difference_style_parameter(loo_data_pair):
    """Test style parameter for plot_loo_difference."""
    loo_result1, loo_result2 = loo_data_pair
    x_values = np.arange(len(loo_result1.loo_i))
    ax = plot_loo_difference(x_values, loo_result1, loo_result2, style="default")
    assert ax is not None

    ax = plot_loo_difference(x_values, loo_result1, loo_result2, style=None)
    assert ax is not None


def test_backend_specification(loo_data):
    """Test specifying backend explicitly."""
    ax = plot_influence(loo_data, backend="matplotlib")
    assert ax is not None

    ax = plot_loo(loo_data, backend="matplotlib")
    assert ax is not None


def test_invalid_backend(loo_data):
    """Test error when an invalid backend is specified."""
    with pytest.raises(KeyError, match="Backend invalid_backend is not implemented"):
        plot_influence(loo_data, backend="invalid_backend")


def test_plot_influence_additional_kwargs(loo_data):
    """Test additional kwargs passed to scatter function in plot_influence."""
    ax = plot_influence(
        loo_data, edgecolors="black", linewidths=1.5, facecolors="white"
    )
    assert ax is not None


def test_plot_loo_additional_kwargs(loo_data):
    """Test additional kwargs passed to scatter function in plot_loo."""
    ax = plot_loo(loo_data, edgecolors="gray", linewidths=2, facecolors="white")
    assert ax is not None


def test_plot_loo_difference_additional_kwargs(loo_data_pair):
    """Test additional kwargs passed to scatter function in plot_loo_difference."""
    loo_result1, loo_result2 = loo_data_pair
    x_values = np.arange(len(loo_result1.loo_i))

    ax = plot_loo_difference(
        x_values,
        loo_result1,
        loo_result2,
        edgecolors="black",
        linewidths=1.5,
        facecolors="white",
    )
    assert ax is not None


def test_plot_influence_show_pareto_k_threshold(loo_data):
    """Test plot_influence with different threshold and k_threshold values."""
    ax = plot_influence(
        loo_data, use_pareto_k=True, k_threshold=0.5, threshold=0.2, n_points=None
    )
    assert ax is not None


def test_plot_loo_difference_with_color_map(loo_data_pair):
    """Test plot_loo_difference with custom colormap."""
    loo_result1, loo_result2 = loo_data_pair
    x_values = np.arange(len(loo_result1.loo_i))

    group_values = np.linspace(0, 1, len(x_values))

    ax = plot_loo_difference(
        x_values, loo_result1, loo_result2, group=group_values, cmap="viridis"
    )
    assert ax is not None


def test_loo_difference_custom_outlier_formatting(loo_data_pair):
    """Test plot_loo_difference with outlier threshold and custom formatting."""
    loo_result1, loo_result2 = loo_data_pair
    x_values = np.arange(len(loo_result1.loo_i))

    ax = plot_loo_difference(
        x_values,
        loo_result1,
        loo_result2,
        outlier_thresh=0.005,
        size=50,
        alpha=0.8,
        edgecolors="red",
        linewidths=2,
    )
    assert ax is not None


def test_multiple_kwargs_together(loo_data):
    """Test combining multiple kwargs on a single plot."""
    ax = plot_influence(
        loo_data,
        color="blue",
        threshold=0.5,
        sort=True,
        n_points=5,
        edgecolors="black",
        linewidths=2,
        facecolors="lightblue",
    )
    assert ax is not None
