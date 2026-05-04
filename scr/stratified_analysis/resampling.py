from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def bca_interval(
    replicates: np.ndarray,
    conf: float = 0.95,
    eps: float = 1e-16,
) -> Tuple[float, float]:

    replicates = np.asarray(replicates, dtype=float)
    replicates = replicates[np.isfinite(replicates)]

    n_replicates = replicates.size

    if n_replicates == 0:
        return np.nan, np.nan

    if n_replicates < 3:
        lo, hi = np.nanpercentile(
            replicates,
            [
                100.0 * (1.0 - conf) / 2.0,
                100.0 * (1.0 + conf) / 2.0,
            ],
        )
        return float(lo), float(hi)

    theta_hat = float(np.nanmean(replicates))

    prop = np.sum(replicates < theta_hat) / float(n_replicates)
    prop = np.clip(prop, eps, 1.0 - eps)
    z0 = norm.ppf(prop)

    theta_jack = np.empty(n_replicates, dtype=float)

    for i in range(n_replicates):
        vals = np.delete(replicates, i)
        theta_jack[i] = np.nanmean(vals) if vals.size > 0 else np.nan

    theta_jack_mean = np.nanmean(theta_jack)
    diffs = theta_jack_mean - theta_jack

    denom = np.sum(diffs**2)

    if denom <= 0 or not np.isfinite(denom):
        lo, hi = np.nanpercentile(
            replicates,
            [
                100.0 * (1.0 - conf) / 2.0,
                100.0 * (1.0 + conf) / 2.0,
            ],
        )
        return float(lo), float(hi)

    num = np.sum(diffs**3)
    acceleration = num / (6.0 * denom**1.5)

    alpha_low = (1.0 - conf) / 2.0
    alpha_high = 1.0 - alpha_low

    def adjusted_quantile(alpha: float) -> float:
        z_alpha = norm.ppf(alpha)
        denom_adj = 1.0 - acceleration * (z0 + z_alpha)

        if denom_adj == 0:
            return float(np.clip(prop, eps, 1.0 - eps))

        adj = norm.cdf(z0 + (z0 + z_alpha) / denom_adj)
        return float(np.clip(adj, eps, 1.0 - eps))

    q_low = adjusted_quantile(alpha_low)
    q_high = adjusted_quantile(alpha_high)

    lo = float(np.percentile(replicates, 100.0 * q_low))
    hi = float(np.percentile(replicates, 100.0 * q_high))

    return lo, hi


def combine_replicates_results(
    all_results: pd.DataFrame,
    alpha: float = 0.05,
    exponentiate: bool = False,
) -> pd.DataFrame:
    """
    Combine per-replicate coefficient estimates into a replicates summary table.

    Expected columns in `all_results`:
    - term
    - coef
    - pval
    """
    required = {"term", "coef", "pval"}
    missing = required - set(all_results.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    def aggregate_term(group: pd.DataFrame) -> pd.Series:
        coefs = np.asarray(group["coef"].to_numpy(), dtype=float)
        pvals = np.asarray(group["pval"].to_numpy(), dtype=float)

        coefs = coefs[np.isfinite(coefs)]
        n_reps = int(coefs.size)

        mean_coef = float(np.nanmean(coefs)) if n_reps > 0 else np.nan

        ci95_low, ci95_high = bca_interval(coefs, conf=0.95)
        ci99_low, ci99_high = bca_interval(coefs, conf=0.99)
        ci999_low, ci999_high = bca_interval(coefs, conf=0.999)

        valid_pmask = np.isfinite(pvals)

        sig_fraction = (
            float(np.mean(pvals[valid_pmask] < alpha)) * 100.0
            if valid_pmask.any()
            else np.nan
        )

        def excludes_zero(lo: float, hi: float) -> bool:
            return (
                not np.isnan(lo)
                and not np.isnan(hi)
                and ((lo > 0) or (hi < 0))
            )

        if excludes_zero(ci999_low, ci999_high):
            significance = "***"
        elif excludes_zero(ci99_low, ci99_high):
            significance = "**"
        elif excludes_zero(ci95_low, ci95_high):
            significance = "*"
        else:
            significance = ""

        out = {
            "mean_coef": mean_coef,
            "CI95_low": ci95_low,
            "CI95_high": ci95_high,
            "CI99_low": ci99_low,
            "CI99_high": ci99_high,
            "CI99.9_low": ci999_low,
            "CI99.9_high": ci999_high,
            "sig_fraction": sig_fraction,
            "significance": significance,
            "n_reps": n_reps,
        }

        if exponentiate:
            exp_cols = {
                "exp_mean": mean_coef,
                "exp_CI95_low": ci95_low,
                "exp_CI95_high": ci95_high,
                "exp_CI99_low": ci99_low,
                "exp_CI99_high": ci99_high,
                "exp_CI99.9_low": ci999_low,
                "exp_CI99.9_high": ci999_high,
            }

            for name, value in exp_cols.items():
                out[name] = float(np.exp(value)) if np.isfinite(value) else np.nan

        return pd.Series(out)

    return (
        all_results.groupby("term", observed=True)
        .apply(aggregate_term)
        .reset_index()
    )