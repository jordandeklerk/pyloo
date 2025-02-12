"""Example demonstrating LOO-CV with a complex hierarchical model of student test scores using PyMC."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

from pyloo.loo import loo
from pyloo.loo_subsample import loo_subsample

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger("student_scores_example")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

np.random.seed(42)

n_schools = 20
n_teachers_per_school = 5
n_subjects = 3
n_students_per_teacher = 30

school_std = 1.0
teacher_std = 0.7
subject_std = 0.5
student_std = 1.2

school_effects = np.random.normal(0, school_std, n_schools)
teacher_effects = np.array([np.random.normal(effect, teacher_std, n_teachers_per_school) for effect in school_effects])
subject_effects = np.random.normal(0, subject_std, n_subjects)

data = []
for school in range(n_schools):
    for teacher_idx in range(n_teachers_per_school):
        teacher_effect = teacher_effects[school][teacher_idx]
        for subject in range(n_subjects):
            student_scores = np.random.normal(
                school_effects[school] + teacher_effect + subject_effects[subject], student_std, n_students_per_teacher
            )
            for score in student_scores:
                data.append(
                    {"school": school, "teacher": f"{school}_{teacher_idx}", "subject": subject, "score": score}
                )

df = pd.DataFrame(data)

with pm.Model() as student_model:
    school_std = pm.HalfNormal("school_std", sigma=2)
    teacher_std = pm.HalfNormal("teacher_std", sigma=2)
    subject_std = pm.HalfNormal("subject_std", sigma=2)

    school_effects = pm.Normal("school_effects", mu=0, sigma=school_std, shape=n_schools)
    subject_effects = pm.Normal("subject_effects", mu=0, sigma=subject_std, shape=n_subjects)
    teacher_effects_raw = pm.Normal("teacher_effects_raw", mu=0, sigma=1, shape=(n_schools, n_teachers_per_school))
    teacher_effects = teacher_std * teacher_effects_raw

    school_idx = df.school.values
    teacher_idx = df.teacher.str.split("_").str[1].astype(int).values
    mu = teacher_effects[school_idx, teacher_idx] + school_effects[school_idx] + subject_effects[df.subject.values]

    scores = pm.Normal("scores", mu=mu, sigma=student_std, observed=df.score.values)

    idata = pm.sample(
        draws=1000,
        tune=500,
        chains=4,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
        target_accept=0.95,
    )

logger.info("\nComputing full LOO-CV...")
loo_results = loo(idata, pointwise=True)
logger.info(loo_results)

logger.info("\nComputing subsampled LOO-CV...")
n_subsample = int(0.2 * len(df))
loo_subsample_results = loo_subsample(idata, observations=n_subsample, loo_approximation="plpd", pointwise=True)
logger.info(loo_subsample_results)

logger.info("\nComparison of estimates:")
logger.info(f"Full LOO ELPD:      {loo_results.elpd_loo:.2f} ± {loo_results.se:.2f}")
logger.info(f"Subsampled LOO ELPD: {loo_subsample_results.elpd_loo:.2f} ± {loo_subsample_results.se:.2f}")

logger.info("\nPareto k diagnostics summary:")
logger.info("Full LOO k values:")
k_values = loo_results.pareto_k
logger.info(f"  Mean: {np.mean(k_values):.3f}")
logger.info(f"  Max:  {np.max(k_values):.3f}")
logger.info(f"  # of k > 0.7: {np.sum(k_values > 0.7)}")

if hasattr(loo_subsample_results, "pareto_k"):
    logger.info("\nSubsampled LOO k values:")
    k_values_sub = loo_subsample_results.pareto_k
    logger.info(f"  Mean: {np.mean(k_values_sub):.3f}")
    logger.info(f"  Max:  {np.max(k_values_sub):.3f}")
    logger.info(f"  # of k > 0.7: {np.sum(k_values_sub > 0.7)}")
