"""Tests for diagnostic functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ...diagnostics import (
    ParetokTable,
    k_cut,
    mcse_loo,
    pareto_k_ids,
    pareto_k_influence_values,
    pareto_k_table,
    pareto_k_values,
    plot_diagnostic,
    ps_khat_threshold,
    psis_ess_values,
)
from ...psis import PSISData, psislw
from ..helpers import assert_arrays_equal


def test_pareto_k_table_basic(log_likelihood_data):
    """Test basic functionality of Pareto k diagnostic table."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    table = pareto_k_table(psis_obj)
    assert isinstance(table, ParetokTable)
    assert len(table.counts) == 3
    assert len(table.proportions) == 3
    assert len(table.min_ess) == 3
    assert 0 <= table.k_threshold <= 1


def test_pareto_k_table_string(log_likelihood_data):
    """Test string representation of Pareto k table."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    table = pareto_k_table(psis_obj)
    table_str = str(table)
    assert isinstance(table_str, str)
    assert "Pareto k diagnostic values:" in table_str


def test_pareto_k_ids_default(log_likelihood_data):
    """Test identification of observations with high Pareto k values using default threshold."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    ids = pareto_k_ids(psis_obj)
    assert isinstance(ids, np.ndarray)
    assert ids.dtype == np.int64


def test_pareto_k_ids_custom(log_likelihood_data):
    """Test identification of observations with high Pareto k values using custom threshold."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    custom_ids = pareto_k_ids(psis_obj, threshold=0.5)
    assert isinstance(custom_ids, np.ndarray)
    assert all(pareto_k[custom_ids] > 0.5)


def test_pareto_k_values_basic(log_likelihood_data):
    """Test basic extraction of Pareto k values."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    k_vals = pareto_k_values(psis_obj)
    assert isinstance(k_vals, np.ndarray)
    assert k_vals.shape == pareto_k.shape
    assert_arrays_equal(k_vals, pareto_k)


def test_pareto_k_values_error():
    """Test error case for Pareto k values extraction."""
    psis_obj = PSISData(log_weights=np.array([]), pareto_k=None)
    with pytest.raises(ValueError, match="No Pareto k estimates found"):
        pareto_k_values(psis_obj)


def test_psis_ess_values_basic(log_likelihood_data):
    """Test basic extraction of PSIS effective sample sizes."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    ess_vals = psis_ess_values(psis_obj)
    assert isinstance(ess_vals, np.ndarray)
    assert ess_vals.shape == ess.shape
    assert_arrays_equal(ess_vals, ess)


def test_psis_ess_values_error():
    """Test error case for PSIS effective sample sizes extraction."""
    psis_obj = PSISData(log_weights=np.array([]), pareto_k=np.array([]))
    with pytest.raises(ValueError, match="No PSIS effective sample size estimates found"):
        psis_ess_values(psis_obj)


def test_mcse_loo_basic(log_likelihood_data):
    """Test basic Monte Carlo standard error computation."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    mcse = np.random.rand(len(pareto_k))
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess, mcse_elpd_loo=mcse)

    error = mcse_loo(psis_obj)
    assert isinstance(error, float)
    assert error >= 0 or np.isnan(error)


def test_mcse_loo_custom_threshold(log_likelihood_data):
    """Test Monte Carlo standard error computation with custom threshold."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    mcse = np.random.rand(len(pareto_k))
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess, mcse_elpd_loo=mcse)

    error = mcse_loo(psis_obj, threshold=0.5)
    assert isinstance(error, float)


def test_mcse_loo_error():
    """Test error case for Monte Carlo standard error computation."""
    psis_obj = PSISData(log_weights=np.array([]), pareto_k=np.array([]))
    with pytest.raises(ValueError, match="Monte Carlo standard error estimates not found"):
        mcse_loo(psis_obj)


def test_plot_diagnostic_k(log_likelihood_data):
    """Test PSIS diagnostic k plot."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    ax = plot_diagnostic(psis_obj, diagnostic="k")
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() == "Pareto shape k"
    plt.close()


def test_plot_diagnostic_ess(log_likelihood_data):
    """Test PSIS diagnostic ess plot."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    ax = plot_diagnostic(psis_obj, diagnostic="ess")
    assert isinstance(ax, plt.Axes)
    assert ax.get_ylabel() == "PSIS ESS"
    plt.close()


def test_plot_diagnostic_options(log_likelihood_data):
    """Test PSIS diagnostic plot options."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    ax = plot_diagnostic(psis_obj, diagnostic="k", label_points=True)
    assert isinstance(ax, plt.Axes)
    plt.close()

    ax = plot_diagnostic(psis_obj, title="Custom Title")
    assert ax.get_title() == "Custom Title"
    plt.close()

    fig, ax = plt.subplots()
    ax_returned = plot_diagnostic(psis_obj, ax=ax)
    assert ax_returned is ax
    plt.close()


def test_k_cut_basic():
    """Test basic Pareto k value categorization."""
    k_values = np.array([-0.5, 0.3, 0.6, 0.8, 1.2])
    threshold = 0.7

    categories = k_cut(k_values, threshold)
    assert isinstance(categories, np.ndarray)
    assert categories.dtype == int
    assert len(categories) == len(k_values)


def test_k_cut_categories():
    """Test Pareto k value category assignments."""
    k_values = np.array([-0.5, 0.3, 0.6, 0.8, 1.2])
    threshold = 0.7

    categories = k_cut(k_values, threshold)
    assert all(categories[k_values <= threshold] == 0)  # good
    assert all(categories[(k_values > threshold) & (k_values <= 1)] == 1)  # bad
    assert all(categories[k_values > 1] == 2)  # very bad


def test_ps_khat_threshold():
    """Test sample size dependent threshold computation."""
    for S in [10, 100, 1000, 10000]:
        threshold = ps_khat_threshold(S)
        assert isinstance(threshold, float)
        assert 0 < threshold <= 0.7

        expected = min(1 - 1 / np.log(S), 0.7)
        assert np.isclose(threshold, expected)


def test_edge_cases(extreme_data):
    """Test diagnostic functions with extreme data."""
    log_weights, pareto_k, ess = psislw(extreme_data)
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess)

    table = pareto_k_table(psis_obj)
    assert isinstance(table, ParetokTable)

    ids = pareto_k_ids(psis_obj)
    assert isinstance(ids, np.ndarray)

    ax = plot_diagnostic(psis_obj)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_pareto_k_influence_values_basic(log_likelihood_data):
    """Test basic extraction of Pareto k influence values."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = psislw(log_ratios)
    influence_k = np.random.rand(len(pareto_k))
    psis_obj = PSISData(log_weights=log_weights, pareto_k=pareto_k, ess=ess, influence_pareto_k=influence_k)

    k_inf = pareto_k_influence_values(psis_obj)
    assert isinstance(k_inf, np.ndarray)
    assert k_inf.shape == influence_k.shape
    assert_arrays_equal(k_inf, influence_k)


def test_pareto_k_influence_values_error():
    """Test error case for Pareto k influence values extraction."""
    psis_obj = PSISData(log_weights=np.array([]), pareto_k=np.array([]))
    with pytest.raises(ValueError, match="No Pareto k influence estimates found"):
        pareto_k_influence_values(psis_obj)
