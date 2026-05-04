from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from stratified_analysis.resampling import combine_replicates_results


warnings.filterwarnings(
    "ignore",
    message="Random effects covariance is singular",
    category=UserWarning,
    module=r"statsmodels\.regression\.mixed_linear_model",
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


USE_RPY2 = False

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    ro.r('library("lme4")')
    ro.r('library("broom.mixed")')
    USE_RPY2 = True
except Exception:
    USE_RPY2 = False
    ro = None


def is_constant_or_nan(series: pd.Series) -> bool:
    """Return True if a series is all NaN or has only one unique non-NaN value."""
    if series.isna().all():
        return True

    return len(series.dropna().unique()) <= 1


def build_interaction_formula(
    y: str,
    fixed_terms: List[str],
    data: pd.DataFrame,
) -> str:
    """Build formula of the form y ~ ap_dur * term1 + ap_dur * term2."""
    terms = [term for term in fixed_terms if term in data.columns]

    if not terms:
        raise RuntimeError("No fixed_terms found in data.")

    interactions = [f"ap_dur * {term}" for term in terms]

    return f"{y} ~ {' + '.join(interactions)}"


def fit_continuous_robust(
    df_rep: pd.DataFrame,
    fixed_terms: List[str],
    y: str,
    group_col: str,
    reml: bool = False,
    tol_singular: float = 1e-8,
    icc_thresh: float = 1e-4,
) -> list[dict[str, float | str]]:
    """
    Fit a continuous outcome model.

    Uses MixedLM when random effect looks meaningful.
    Falls back to OLS with cluster-robust covariance when needed.
    """
    if is_constant_or_nan(df_rep[y]):
        raise RuntimeError("Outcome constant or all-NaN in replicate.")

    df_work = df_rep.copy()
    formula = build_interaction_formula(y=y, fixed_terms=fixed_terms, data=df_work)

    try:
        grouped = df_work.groupby(group_col)[y]
        between_var = grouped.mean().var(ddof=1) if grouped.size().size > 0 else 0.0
        total_var = df_work[y].var(ddof=1)

        icc = (
            between_var / total_var
            if total_var is not None and np.isfinite(total_var) and total_var != 0
            else 0.0
        )
    except Exception:
        icc = 0.0

    num_groups = df_work[group_col].nunique(dropna=True)

    if num_groups < 3 or icc < icc_thresh:
        return fit_ols_cluster_robust(df_work, formula, group_col)

    try:
        model = smf.mixedlm(
            formula,
            df_work,
            groups=df_work[group_col],
            re_formula="1",
        )

        result = model.fit(
            reml=reml,
            method="lbfgs",
            maxiter=200,
            disp=False,
        )

        try:
            cov_re = np.asarray(result.cov_re)
            eigs = np.linalg.eigvalsh(cov_re)

            if np.any(eigs < tol_singular):
                raise RuntimeError("Random effects covariance near-singular.")

        except Exception:
            pass

        return [
            {
                "term": str(term),
                "coef": float(result.params[term]),
                "pval": float(result.pvalues[term]),
            }
            for term in result.params.index.astype(str)
        ]

    except Exception:
        return fit_ols_cluster_robust(df_work, formula, group_col)


def fit_ols_cluster_robust(
    data: pd.DataFrame,
    formula: str,
    group_col: str,
) -> list[dict[str, float | str]]:
    """Fit OLS with cluster-robust covariance."""
    ols = sm.OLS.from_formula(formula, data=data).fit()

    clusters = data[group_col]
    cov = sm.stats.sandwich_covariance.cov_cluster(ols, clusters)

    params = ols.params
    bse = np.sqrt(np.diag(cov))

    z_values = params / bse
    pvals = 2.0 * norm.sf(np.abs(z_values))

    return [
        {
            "term": str(term),
            "coef": float(params[term]),
            "pval": float(pvals[i]),
        }
        for i, term in enumerate(params.index.astype(str))
    ]


def fit_binary_robust(
    rep_df: pd.DataFrame,
    fixed_terms: List[str],
    y: str,
    group_col: str,
) -> list[dict[str, float | str]]:
    """
    Fit binary response model.

    Uses R glmer if rpy2/lme4 are available.
    Otherwise falls back to statsmodels GLM with binomial family.
    """
    if is_constant_or_nan(rep_df[y]):
        raise RuntimeError("Binary outcome constant or all-NaN in replicate.")

    df_work = rep_df.copy()
    formula_py = build_interaction_formula(y=y, fixed_terms=fixed_terms, data=df_work)

    terms = [term for term in fixed_terms if term in df_work.columns]
    fixed_formula = " + ".join([f"ap_dur * {term}" for term in terms])
    formula_r = f"{y} ~ {fixed_formula} + (1 | {group_col})"

    if USE_RPY2:
        from rpy2.robjects import pandas2ri as p2r

        p2r.activate()
        ro.globalenv["tmpdf"] = p2r.py2rpy(df_work)

        r_code = f"""
        library(lme4)
        library(broom.mixed)

        fit <- try(
            glmer(
                {formula_r},
                data = tmpdf,
                family = binomial(link = "logit"),
                control = glmerControl(
                    optimizer = "bobyqa",
                    optCtrl = list(maxfun = 200000)
                )
            ),
            silent = TRUE
        )

        if (inherits(fit, "try-error")) {{
            NULL
        }} else {{
            broom.mixed::tidy(
                fit,
                effects = "fixed",
                conf.int = FALSE
            )[, c("term", "estimate", "p.value")]
        }}
        """

        result = ro.r(r_code)

        if result is None or len(result) == 0:
            raise RuntimeError("R glmer returned no result.")

        r_out = p2r.rpy2py(result)

        return [
            {
                "term": str(row["term"]),
                "coef": float(row["estimate"]),
                "pval": float(row["p.value"]),
            }
            for _, row in r_out.iterrows()
        ]

    glm = sm.GLM.from_formula(
        formula_py,
        data=df_work,
        family=sm.families.Binomial(),
    )

    glm_result = glm.fit()

    try:
        robust = glm_result.get_robustcov_results(
            cov_type="cluster",
            groups=df_work[group_col],
        )
        params = robust.params
        pvals = robust.pvalues
    except Exception:
        params = glm_result.params
        pvals = getattr(
            glm_result,
            "pvalues",
            pd.Series(np.nan, index=params.index),
        )

    return [
        {
            "term": str(term),
            "coef": float(params[term]),
            "pval": float(pvals[term]),
        }
        for term in params.index.astype(str)
    ]


def fit_mixed_models_on_bootstraps(
    replicates: List[pd.DataFrame],
    fixed_terms: List[str],
    y: str,
    random_effect: str = "ID",
    alpha: float = 0.05,
    reml: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fit model to each bootstrap replicate and combine coefficient summaries."""
    all_results = []

    for i, rep in enumerate(replicates):
        if rep.empty:
            continue

        try:
            if y == "response":
                result_list = fit_binary_robust(
                    rep_df=rep,
                    fixed_terms=fixed_terms,
                    y=y,
                    group_col=random_effect,
                )
            else:
                result_list = fit_continuous_robust(
                    df_rep=rep,
                    fixed_terms=fixed_terms,
                    y=y,
                    group_col=random_effect,
                    reml=reml,
                )

            for result in result_list:
                all_results.append(
                    {
                        "term": result["term"],
                        "coef": result["coef"],
                        "pval": result["pval"],
                        "replicate": i,
                    }
                )

        except Exception as exc:
            if verbose:
                print(f"[mixed model] replicate={i}, y={y} failed: {exc}")

    if not all_results:
        return pd.DataFrame(
            columns=[
                "term",
                "mean_coef",
                "CI95_low",
                "CI95_high",
                "CI99_low",
                "CI99_high",
                "CI99.9_low",
                "CI99.9_high",
                "sig_fraction",
                "significance",
                "n_reps",
            ]
        )

    all_results_df = pd.DataFrame(all_results)

    return combine_replicates_results(
        all_results=all_results_df,
        alpha=alpha,
        exponentiate=(y == "response"),
    )