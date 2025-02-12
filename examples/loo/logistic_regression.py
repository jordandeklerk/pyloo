"""Example demonstrating LOO-CV with a logistic regression model using CmdStanPy."""

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from cmdstanpy import CmdStanModel

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from pyloo.loo import loo  # noqa: E402
from pyloo.loo_subsample import loo_subsample  # noqa: E402
from pyloo.utils import to_inference_data  # noqa: E402

logger = logging.getLogger("logistic_regression_example")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

STAN_MODEL = """
data {
  int<lower=0> N;
  int<lower=0> P;
  matrix[N,P] X;
  array[N] int<lower=0,upper=1> y;
}
parameters {
  vector[P] beta;
}
model {
  beta ~ normal(0, 5);

  y ~ bernoulli_logit(X * beta);
}
generated quantities {
  vector[N] log_lik;
  {
    vector[N] eta = X * beta;
    for (n in 1:N) {
      log_lik[n] = y[n] * eta[n] - log1p_exp(eta[n]);
    }
  }
}
"""


def generate_synthetic_data(n_samples: int = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data similar to the wells dataset but larger."""
    rng = np.random.default_rng(seed)

    dist = rng.uniform(0, 200, n_samples)
    dist100 = dist / 100
    arsenic = rng.lognormal(0, 1, n_samples)

    true_beta = np.array([0.5, -0.4, 0.8])

    X = np.column_stack(
        [np.ones(n_samples), (dist100 - dist100.mean()) / dist100.std(), (arsenic - arsenic.mean()) / arsenic.std()]
    )

    logits = X @ true_beta
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(n=1, p=probs)

    return X, y


def main():
    """Run the logistic regression example."""
    n_samples = 10000
    logger.info("Generating synthetic data with %d samples...", n_samples)
    X, y = generate_synthetic_data(n_samples=n_samples)

    stan_data = {
        "N": X.shape[0],
        "P": X.shape[1],
        "X": X,
        "y": y,
    }

    model_path = Path("logistic_model.stan")
    model_path.write_text(STAN_MODEL)

    logger.info("Compiling and fitting Stan model...")
    model = CmdStanModel(stan_file=str(model_path))
    fit = model.sample(data=stan_data, chains=4, iter_sampling=2000, iter_warmup=1000, seed=42)

    logger.info("Converting to InferenceData format...")
    idata = to_inference_data(fit)

    logger.info("\nComputing full LOO-CV...")
    loo_results = loo(idata, pointwise=True)
    logger.info(loo_results)

    logger.info("\nComputing subsampled LOO-CV...")
    n_subsample = 400
    loo_subsample_results = loo_subsample(idata, observations=n_subsample, loo_approximation="plpd", pointwise=True)
    logger.info(loo_subsample_results)

    logger.info("\nComparison of estimates:")
    logger.info(f"Full LOO ELPD:      {loo_results.elpd_loo:.2f} ± {loo_results.se:.2f}")
    logger.info(f"Subsampled LOO ELPD: {loo_subsample_results.elpd_loo:.2f} ± {loo_subsample_results.se:.2f}")

    logger.info("\nPareto k diagnostics:")
    k_values = loo_results.pareto_k
    logger.info("Full LOO k values:")
    logger.info(f"  Mean: {np.mean(k_values):.3f}")
    logger.info(f"  Max:  {np.max(k_values):.3f}")
    logger.info(f"  # of k > 0.7: {np.sum(k_values > 0.7)}")

    if hasattr(loo_subsample_results, "pareto_k"):
        k_values_sub = loo_subsample_results.pareto_k
        logger.info("\nSubsampled LOO k values:")
        logger.info(f"  Mean: {np.mean(k_values_sub):.3f}")
        logger.info(f"  Max:  {np.max(k_values_sub):.3f}")
        logger.info(f"  # of k > 0.7: {np.sum(k_values_sub > 0.7)}")

    if model_path.exists():
        model_path.unlink()


if __name__ == "__main__":
    main()
