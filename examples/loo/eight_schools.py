"""Example demonstrating LOO-CV with the Eight Schools model using PyMC."""

import logging
import sys
from pathlib import Path

import numpy as np
import pymc as pm

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from pyloo.compare import loo_compare  # noqa: E402
from pyloo.loo import loo  # noqa: E402
from pyloo.loo_subsample import loo_subsample  # noqa: E402

logger = logging.getLogger("eight_schools_example")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

schools = 8
treatment_effects = np.array([28, 8, -3, 7, -1, 1, 18, 12])
treatment_stddevs = np.array([15, 10, 16, 11, 9, 11, 10, 18])

with pm.Model() as eight_schools:
    mu = pm.Normal("mu", mu=0, sigma=10)
    tau = pm.HalfCauchy("tau", beta=5)

    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=schools)
    y = pm.Normal("y", mu=theta, sigma=treatment_stddevs, observed=treatment_effects)

    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )


with pm.Model() as eight_schools_2:
    mu = pm.Normal("mu", mu=0, sigma=10)
    tau = pm.InverseGamma("tau", alpha=2, beta=5)

    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=schools)
    y = pm.Normal("y", mu=theta, sigma=treatment_stddevs, observed=treatment_effects)

    idata_2 = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )

logger.info("\nComputing full LOO-CV...")
loo_results = loo(idata, pointwise=True)
logger.info(loo_results)

logger.info("\nComputing subsampled LOO-CV...")
loo_results_2 = loo(idata_2, pointwise=True)
logger.info(loo_results_2)

logger.info("\nComputing subsampled LOO-CV...")
loo_subsample_results = loo_subsample(idata, observations=4, pointwise=True)
logger.info(loo_subsample_results)

logger.info("\nComputing model comparison...")
comparison_results = loo_compare({"model1": loo_results, "model2": loo_results_2})
logger.info(comparison_results)

logger.info("\nComputing model comparison with Bayesian bootstrap...")
comparison_results_bb = loo_compare(
    {"model1": loo_results, "model2": loo_results_2},
    method="bb-pseudo-bma",
    alpha=0.3,
    b_samples=1000,
)
logger.info(comparison_results_bb)

logger.info("\nComparison of estimates:")
logger.info(f"Full LOO ELPD:      {loo_results.elpd_loo:.2f} ± {loo_results.se:.2f}")
logger.info(f"Subsampled LOO ELPD: {loo_subsample_results.elpd_loo:.2f} ± {loo_subsample_results.se:.2f}")

logger.info("\nPareto k diagnostics:")
logger.info("Full LOO k values: %s", loo_results.pareto_k)
if hasattr(loo_subsample_results, "pareto_k"):
    logger.info("Subsampled LOO k values: %s", loo_subsample_results.pareto_k)
