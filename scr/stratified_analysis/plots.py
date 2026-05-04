from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from stratified_analysis.labels import (
    remove_top_right_and_grid,
    y_label_from_col,
)
from stratified_analysis.windows import (
    aggregate_by_windows_for_centers,
    compute_apdur_split_curves,
    make_window_centers,
    merge_sparse_windows,
)


def compute_feature_combined_across_replicates(
    replicates: List[pd.DataFrame],
    feat_col: str,
    kind: str,
    n_windows_feat: int = 40,
    window_overlap_feat: float = 0.5,
    id_col: str = "ID",
    y_col: str = "response",
    min_points: int = 20,
    min_unique_ids: int = 3,
    labels: Optional[List[str]] = None,
    max_merge_size: int = 10,
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """
    Compute feature-level aggregated outcome curves across bootstrap replicates.

    For continuous features:
        Uses sliding windows and merges sparse adjacent windows.

    For categorical features:
        Computes category-level means across replicates.

    Returns
    -------
    x_values_or_labels, (mean, lo, hi)
    """
    if not replicates:
        return None, (None, None, None)

    merged_feat = pd.concat(replicates, ignore_index=True)

    if feat_col not in merged_feat.columns:
        return None, (None, None, None)

    if kind == "cont":
        x_min = float(merged_feat[feat_col].min())
        x_max = float(merged_feat[feat_col].max())

        centers, width = make_window_centers(
            x_min=x_min,
            x_max=x_max,
            n_windows=n_windows_feat,
            overlap=window_overlap_feat,
            explicit_width=None,
        )

        centers, widths = merge_sparse_windows(
            centers=centers,
            width=width,
            data=merged_feat,
            x_col=feat_col,
            id_col=id_col,
            min_points=min_points,
            min_unique_ids=min_unique_ids,
            max_merge_size=max_merge_size,
        )

        n_centers = len(centers)

        per_rep = []

        for rep in replicates:
            if rep.empty:
                per_rep.append(np.full(n_centers, np.nan))
                continue

            values, _, _ = aggregate_by_windows_for_centers(
                data=rep,
                x_col=feat_col,
                centers=centers,
                window_width=widths,
                y_col=y_col,
                id_col=id_col,
                min_points=min_points,
                min_unique_ids=min_unique_ids,
            )
            per_rep.append(values)

        matrix = np.vstack(per_rep)

        mean = np.nanmean(matrix, axis=0)
        lo = np.nanpercentile(matrix, 2.5, axis=0)
        hi = np.nanpercentile(matrix, 97.5, axis=0)

        return centers, (mean, lo, hi)

    cats = sorted(merged_feat[feat_col].dropna().unique().tolist())

    if not cats:
        return None, (None, None, None)

    per_rep = []

    for rep in replicates:
        if rep.empty:
            per_rep.append(np.full(len(cats), np.nan))
            continue

        grouped = rep.groupby(feat_col, observed=True)[y_col]
        means = grouped.mean().reindex(cats, fill_value=np.nan).to_numpy()

        if y_col == "response":
            values = 100.0 * means
        else:
            values = means.astype(float)

        values[~np.isfinite(values)] = np.nan
        per_rep.append(values)

    matrix = np.vstack(per_rep)

    mean = np.nanmean(matrix, axis=0)
    lo = np.nanpercentile(matrix, 2.5, axis=0)
    hi = np.nanpercentile(matrix, 97.5, axis=0)

    if labels is not None and len(labels) == len(cats):
        display_labels = labels
    else:
        display_labels = [str(cat) for cat in cats]

    return display_labels, (mean, lo, hi)


def make_group_functions_for_feature(
    replicates: List[pd.DataFrame],
    feat_kind: str,
    feat_kwargs: Dict[str, Any],
) -> Tuple[List[Callable[[pd.DataFrame], pd.DataFrame]], List[str]]:
    """
    Build subgroup functions and subgroup labels for a feature.

    Continuous features are split by thresholds.
    Categorical features are split by category values.
    """
    if not replicates:
        return [], []

    col = feat_kwargs.get("x_col")
    thresholds = feat_kwargs.get("thr", None)
    labels = feat_kwargs.get("labels", None)

    merged = pd.concat(replicates, ignore_index=True)

    group_fns: List[Callable[[pd.DataFrame], pd.DataFrame]] = []
    group_labels: List[str] = []

    if col not in merged.columns:
        return group_fns, group_labels

    if feat_kind == "cont":
        if thresholds is None or len(thresholds) == 0:
            median = merged[col].median()
            thresholds_sorted = [median]
        else:
            thresholds_sorted = sorted([t for t in thresholds if pd.notna(t)])

        bounds = [-np.inf] + thresholds_sorted + [np.inf]

        for i in range(len(bounds) - 1):
            low = bounds[i]
            high = bounds[i + 1]

            if np.isneginf(low):

                def fn(df: pd.DataFrame, c: str = col, h: float = high) -> pd.DataFrame:
                    return df.loc[df[c] < h]

            elif np.isposinf(high):

                def fn(df: pd.DataFrame, c: str = col, l: float = low) -> pd.DataFrame:
                    return df.loc[df[c] >= l]

            else:

                def fn(
                    df: pd.DataFrame,
                    c: str = col,
                    l: float = low,
                    h: float = high,
                ) -> pd.DataFrame:
                    return df.loc[(df[c] >= l) & (df[c] < h)]

            group_fns.append(fn)

        if labels is not None and len(labels) == len(group_fns):
            group_labels = labels
        else:
            for i in range(len(bounds) - 1):
                low = bounds[i]
                high = bounds[i + 1]

                if np.isneginf(low):
                    group_labels.append(f"< {high}")
                elif np.isposinf(high):
                    group_labels.append(f"≥ {low}")
                else:
                    group_labels.append(f"[{low}, {high})")

    else:
        if thresholds is None or len(thresholds) == 0:
            cats = sorted(merged[col].dropna().unique().tolist())
        else:
            cats = thresholds

        for cat in cats:

            def fn(df: pd.DataFrame, c: str = col, v: Any = cat) -> pd.DataFrame:
                return df.loc[df[c] == v]

            group_fns.append(fn)

        if labels is not None and len(labels) == len(group_fns):
            group_labels = labels
        else:
            group_labels = [str(cat) for cat in cats]

    return group_fns, group_labels


def generate_feature_plot_data(
    replicates: List[pd.DataFrame],
    rows: List[Tuple],
    feat_kind: str,
    feat_kwargs: Dict[str, Any],
    y_col: str = "response",
    n_windows_apdur: int = 40,
    window_overlap_apdur: float = 0.5,
    n_windows_feat: int = 40,
    window_overlap_feat: float = 0.5,
    min_points: int = 20,
    min_unique_ids: int = 3,
    id_col: str = "ID",
    max_merge_size: int = 10,
) -> Dict[str, Any]:
    """
    Generate plot-ready data for one feature.

    Returns data for:
    - feature-vs-outcome plot
    - apnoea-duration subgroup plot
    """
    col = feat_kwargs.get("x_col")
    x_label = feat_kwargs.get("x_label", col)
    labels = feat_kwargs.get("labels", None)

    group_fns, group_labels = make_group_functions_for_feature(
        replicates=replicates,
        feat_kind=feat_kind,
        feat_kwargs=feat_kwargs,
    )

    centers_ap, apdur_results = compute_apdur_split_curves(
        replicates=replicates,
        group_fns=group_fns,
        n_windows_apdur=n_windows_apdur,
        window_overlap_apdur=window_overlap_apdur,
        window_width_apdur=None,
        id_col=id_col,
        y_col=y_col,
        min_points=min_points,
        min_unique_ids=min_unique_ids,
        max_merge_size=max_merge_size,
    )

    feature_x, feature_result = compute_feature_combined_across_replicates(
        replicates=replicates,
        feat_col=col,
        kind=feat_kind,
        n_windows_feat=n_windows_feat,
        window_overlap_feat=window_overlap_feat,
        id_col=id_col,
        y_col=y_col,
        min_points=min_points,
        min_unique_ids=min_unique_ids,
        labels=labels,
        max_merge_size=max_merge_size,
    )

    return {
        "feature": col,
        "x_label": x_label,
        "kind": feat_kind,
        "feature_plot": (feature_x, feature_result),
        "apdur_plot": (centers_ap, apdur_results, group_labels),
    }


def save_single_apdur_plot(
    replicates: List[pd.DataFrame],
    y_col: str,
    out_path: str | Path,
    id_col: str = "ID",
    n_windows_apdur: int = 80,
    window_overlap_apdur: float = 0.5,
    window_width_apdur: Optional[float] = None,
    min_points: int = 20,
    min_unique_ids: int = 3,
    max_merge_size: int = 10,
    figsize: Tuple[float, float] = (6, 3),
    dpi: int = 300,
) -> str:
    """
    Save a single apnoea-duration plot for one outcome.

    Saves:
    - PNG
    - PDF

    Returns
    -------
    Base path without extension.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    centers_ap, apdur_results = compute_apdur_split_curves(
        replicates=replicates,
        group_fns=[lambda df: df],
        n_windows_apdur=n_windows_apdur,
        window_overlap_apdur=window_overlap_apdur,
        window_width_apdur=window_width_apdur,
        id_col=id_col,
        y_col=y_col,
        min_points=min_points,
        min_unique_ids=min_unique_ids,
        max_merge_size=max_merge_size,
    )

    mean, lo, hi = apdur_results[0]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(centers_ap, mean, color="C0", linewidth=1.2)

    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.fill_between(centers_ap, lo, hi, alpha=0.18, color="C0")

    ax.set_xlabel("Apnoea duration (seconds)", fontsize=8)
    ax.set_ylabel(y_label_from_col(y_col), fontsize=8)
    ax.tick_params(labelsize=8)

    remove_top_right_and_grid(ax)

    base = out_path / f"{y_col}_apdur_all"

    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)

    return str(base)


def _format_pearson_p_value(p_value: float) -> str:
    """Format Pearson p-values for compact plot annotations."""
    if not np.isfinite(p_value):
        return "NA"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.3f}"


def annotate_pearson_on_axis(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    fontsize: int = 7,
) -> Tuple[float, float]:
    """
    Compute and annotate Pearson correlation on an axis.

    Returns
    -------
    r, p
    """
    try:
        mask = np.isfinite(x) & np.isfinite(y)

        if (
            np.sum(mask) >= 2
            and np.nanstd(x[mask]) > 0
            and np.nanstd(y[mask]) > 0
        ):
            r_value, p_value = pearsonr(x[mask], y[mask])
            label = f"r = {r_value:.2f}\np = {_format_pearson_p_value(p_value)}"
        else:
            r_value, p_value = np.nan, np.nan
            label = "r = NA\np = NA"

    except Exception:
        r_value, p_value = np.nan, np.nan
        label = "r = NA\np = NA"

    ax.text(
        0.98,
        0.95,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
    )

    return float(r_value), float(p_value)


def save_three_apdur_plot(
    replicates: List[pd.DataFrame],
    ys: List[str],
    out_path: str | Path,
    id_col: str = "ID",
    n_windows_apdur: int = 80,
    window_overlap_apdur: float = 0.5,
    window_width_apdur: Optional[float] = None,
    min_points: int = 20,
    min_unique_ids: int = 3,
    max_merge_size: int = 10,
    figsize: Tuple[float, float] = (9, 4),
    dpi: int = 300,
) -> str:
    """
    Save a combined apnoea-duration plot for multiple outcomes.

    Behaviour:
    - Places `response` in the right-most panel if present.
    - Uses fixed x-axis ticks at 10, 20, 30, 40 seconds when in range.
    - Adds panel labels A), B), C), ...
    - Adds Pearson r and p annotation to each panel.

    Returns
    -------
    Base path without extension.
    """
    if not replicates:
        return ""

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    ys_ordered = [y for y in ys if y != "response"]

    if "response" in ys:
        ys_ordered.append("response")

    results_per_y = []

    for y_col in ys_ordered:
        centers_ap, apdur_results = compute_apdur_split_curves(
            replicates=replicates,
            group_fns=[lambda df: df],
            n_windows_apdur=n_windows_apdur,
            window_overlap_apdur=window_overlap_apdur,
            window_width_apdur=window_width_apdur,
            id_col=id_col,
            y_col=y_col,
            min_points=min_points,
            min_unique_ids=min_unique_ids,
            max_merge_size=max_merge_size,
        )

        mean, lo, hi = apdur_results[0]
        results_per_y.append((y_col, centers_ap, mean, lo, hi))

    n_panels = len(results_per_y)

    if n_panels == 0:
        return ""

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        squeeze=False,
    )

    panel_labels = [
        "A)",
        "B)",
        "C)",
        "D)",
        "E)",
        "F)",
        "G)",
        "H)",
    ]

    for i, (y_col, centers_ap, mean, lo, hi) in enumerate(results_per_y):
        ax = axes[0, i]

        panel_label = panel_labels[i] if i < len(panel_labels) else f"{i + 1})"

        ax.text(
            0.02,
            1.1,
            panel_label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

        if centers_ap is None or len(centers_ap) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            remove_top_right_and_grid(ax)
            continue

        ax.plot(centers_ap, mean, color="C0", linewidth=1.2)

        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(centers_ap, lo, hi, alpha=0.18, color="C0")

        ax.set_xlabel("Apnoea duration (seconds)", fontsize=8)
        ax.set_ylabel(y_label_from_col(y_col), fontsize=8)
        ax.tick_params(labelsize=8)

        finite_mask = (
            np.isfinite(mean)
            | np.isfinite(lo)
            | np.isfinite(hi)
        )

        if finite_mask.any():
            first_idx = int(np.argmax(finite_mask))
            last_idx = int(len(finite_mask) - 1 - np.argmax(finite_mask[::-1]))

            x_min_plot = float(centers_ap[first_idx])
            x_max_plot = float(centers_ap[last_idx])

            x_range = x_max_plot - x_min_plot
            pad = 0.025 * x_range if x_range > 0 else 0.5

            ax.set_xlim((x_min_plot - pad, x_max_plot + pad))
        else:
            ax.set_xlim((np.nanmin(centers_ap), np.nanmax(centers_ap)))

        desired_ticks = np.array([10, 20, 30, 40], dtype=float)
        xlim = ax.get_xlim()

        ticks = desired_ticks[
            (desired_ticks >= xlim[0])
            & (desired_ticks <= xlim[1])
        ]

        if ticks.size > 0:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=8)
        else:
            mid = 0.5 * sum(xlim)
            ax.set_xticks([round(mid)])
            ax.set_xticklabels([f"{round(mid)}"], fontsize=8)

        annotate_pearson_on_axis(
            ax=ax,
            x=np.asarray(centers_ap, dtype=float),
            y=np.asarray(mean, dtype=float),
            fontsize=7,
        )

        remove_top_right_and_grid(ax)

    plt.tight_layout()

    base = out_path / "apdur_all_3panel"

    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)

    return str(base)


def _finite_ylim_from_curves(
    curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    fallback: Tuple[float, float] = (0.0, 1.0),
    pad_fraction: float = 0.05,
) -> Tuple[float, float]:
    """Compute shared y-limits from a list of mean/lo/hi curves."""
    lows = []
    highs = []

    for _, lo, hi in curves:
        if lo is not None:
            finite_lo = lo[np.isfinite(lo)]
            if finite_lo.size > 0:
                lows.append(np.nanmin(finite_lo))

        if hi is not None:
            finite_hi = hi[np.isfinite(hi)]
            if finite_hi.size > 0:
                highs.append(np.nanmax(finite_hi))

    if not lows or not highs:
        return fallback

    y_low = float(np.nanmin(lows))
    y_high = float(np.nanmax(highs))

    y_range = y_high - y_low

    if y_range == 0:
        y_range = abs(y_high) * 0.1 if y_high != 0 else 1.0

    return (
        y_low - pad_fraction * y_range,
        y_high + pad_fraction * y_range,
    )


def _plot_continuous_feature_panel(
    ax: plt.Axes,
    xvals: np.ndarray,
    mean: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    x_label: str,
    y_col: str,
    y_lim: Tuple[float, float],
    shade: bool = False,
) -> None:
    """Plot a continuous feature panel."""
    ax.plot(xvals, mean, color="C0", linewidth=1.0)

    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.fill_between(xvals, lo, hi, alpha=0.18, color="C0")

    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label_from_col(y_col), fontsize=8)

    try:
        ax.set_xlim((np.nanmin(xvals), np.nanmax(xvals)))
    except Exception:
        pass

    ax.set_ylim(y_lim)
    ax.tick_params(labelsize=8)

    if shade:
        try:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.fill_between(
                [x0, x1],
                y0,
                y1,
                color="lightgrey",
                alpha=0.3,
                zorder=0,
            )
        except Exception:
            pass


def _plot_categorical_feature_panel(
    ax: plt.Axes,
    xvals: List[str],
    mean: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    feature: str,
    x_label: str,
    y_col: str,
    y_lim: Tuple[float, float],
    categorical_point_features: set[str],
    shade: bool = False,
) -> None:
    """Plot a categorical feature panel."""
    xt = np.arange(len(xvals))

    if feature in categorical_point_features:
        lower_err = mean - lo
        upper_err = hi - mean

        lower_err = np.where(np.isfinite(lower_err), lower_err, 0.0)
        upper_err = np.where(np.isfinite(upper_err), upper_err, 0.0)

        y_err = np.vstack([lower_err, upper_err])

        ax.errorbar(
            xt,
            mean,
            yerr=y_err,
            fmt="_",
            ecolor="C0",
            elinewidth=1.2,
            capsize=3,
        )
    else:
        ax.plot(xt, mean, marker="o", linestyle="-", color="C0")

        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(xt, lo, hi, alpha=0.18, color="C0")

    ax.set_xticks(xt)
    ax.set_xticklabels([str(label) for label in xvals], rotation=0, fontsize=8)

    ax.set_xlim((-0.5, max(len(xvals) - 0.5, -0.5)))
    ax.set_ylim(y_lim)

    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label_from_col(y_col), fontsize=8)
    ax.tick_params(labelsize=8)

    if shade:
        try:
            ax.axvspan(
                -0.5,
                len(xvals) - 0.5,
                color="lightgrey",
                alpha=0.3,
                zorder=0,
            )
        except Exception:
            pass


def _plot_apdur_subgroup_panel(
    ax: plt.Axes,
    centers: np.ndarray,
    ap_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    subgroup_labels: List[str],
    y_col: str,
    y_lim: Tuple[float, float],
    legend_title: str,
    feature: str,
    shade: bool = False,
) -> None:
    """Plot apnoea-duration subgroup curves."""
    palette = ["C0", "C1", "C2", "C3"]

    for j, (mean, lo, hi) in enumerate(ap_results):
        color = palette[j % len(palette)]

        label = (
            subgroup_labels[j]
            if subgroup_labels is not None and j < len(subgroup_labels)
            else f"g{j}"
        )

        ax.plot(
            centers,
            mean,
            color=color,
            linewidth=1.0,
            label=label,
        )

        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(
                centers,
                lo,
                hi,
                alpha=0.12,
                color=color,
            )

    ax.set_xlabel("Apnoea duration (seconds)", fontsize=8)
    ax.set_ylabel(y_label_from_col(y_col), fontsize=8)
    ax.set_ylim(y_lim)
    ax.tick_params(labelsize=8)

    desired_ticks = np.array([10, 20, 30, 40], dtype=float)
    x0, x1 = ax.get_xlim()

    ticks = desired_ticks[
        (desired_ticks >= x0)
        & (desired_ticks <= x1)
    ]

    if ticks.size > 0:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=8)
    else:
        mid = 0.5 * (x0 + x1)
        ax.set_xticks([round(mid)])
        ax.set_xticklabels([str(round(mid))], fontsize=8)

    if shade:
        try:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            ax.fill_between(
                [x0, x1],
                y0,
                y1,
                color="lightgrey",
                alpha=0.3,
                zorder=0,
            )
        except Exception:
            pass

    if len(ap_results) <= 4:
        if feature == "last_ap_tdiff":
            legend_title = "Time since\nlast apnoea (s)"
        elif feature == "ap_rate_5min":
            legend_title = "Number of apnoea\nin last 5 mins"

        ax.legend(
            frameon=False,
            fontsize=8,
            title=legend_title,
            title_fontsize=8,
            loc="best",
        )


def save_big_figure_for_group(
    replicates: List[pd.DataFrame],
    rows: List[Tuple],
    feature_list: List[str],
    y_col: str,
    out_path: str | Path,
    group_name: str,
    id_col: str = "ID",
    n_windows_apdur: int = 40,
    window_overlap_apdur: float = 0.5,
    n_windows_feat: int = 40,
    window_overlap_feat: float = 0.5,
    min_points: int = 20,
    min_unique_ids: int = 3,
    max_merge_size: int = 10,
    figsize_per_panel: Tuple[float, float] = (3.0, 3.0),
    dpi: int = 300,
) -> Optional[str]:
    """
    Save large grouped feature figure.

    Layout:
    - 2 features per row
    - each feature gets 2 panels:
        left: feature vs outcome
        right: apnoea duration subgroup curves
    - therefore 4 columns per row.

    Returns
    -------
    Base path without extension, or None if no valid features.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    feat_map = {
        row[1]["x_col"]: (row[0], row[1], row[2])
        for row in rows
        if len(row) >= 2 and isinstance(row[1], dict) and "x_col" in row[1]
    }

    selected = []

    for feat in feature_list:
        if feat in feat_map:
            kind, kwargs, _ = feat_map[feat]
            selected.append((kind, kwargs, feat))

    if not selected:
        return None

    all_feature_plots = []
    all_apdur_plots = []

    for kind, kwargs, feat in selected:
        plot_data = generate_feature_plot_data(
            replicates=replicates,
            rows=rows,
            feat_kind=kind,
            feat_kwargs=kwargs,
            y_col=y_col,
            n_windows_apdur=n_windows_apdur,
            window_overlap_apdur=window_overlap_apdur,
            n_windows_feat=n_windows_feat,
            window_overlap_feat=window_overlap_feat,
            min_points=min_points,
            min_unique_ids=min_unique_ids,
            id_col=id_col,
            max_merge_size=max_merge_size,
        )

        feature_x, (feature_mean, feature_lo, feature_hi) = plot_data["feature_plot"]
        ap_centers, ap_results, subgroup_labels = plot_data["apdur_plot"]

        all_feature_plots.append(
            (
                feat,
                feature_x,
                feature_mean,
                feature_lo,
                feature_hi,
                kwargs.get("labels", None),
                kind,
                kwargs.get("x_label", feat),
            )
        )

        all_apdur_plots.append(
            (
                feat,
                ap_centers,
                ap_results,
                subgroup_labels,
                kwargs.get("x_label", feat),
            )
        )

    left_curves = []

    for _, _, mean, lo, hi, _, _, _ in all_feature_plots:
        if mean is not None:
            left_curves.append((mean, lo, hi))

    right_curves = []

    for _, _, ap_results, _, _ in all_apdur_plots:
        for mean, lo, hi in ap_results:
            right_curves.append((mean, lo, hi))

    left_ylim = _finite_ylim_from_curves(left_curves)
    right_ylim = _finite_ylim_from_curves(right_curves)

    n_features = len(all_feature_plots)
    nrows = (n_features + 1) // 2
    ncols = 4

    cm_to_inches = 1.0 / 2.54
    fig_width_inches = 23.0 * cm_to_inches
    fig_height_inches = max(1.0, nrows * figsize_per_panel[1])

    categorical_point_features = {
        "Ventilation",
        "Sex",
        "caffeine",
        "infection",
    }

    left_shade_features = {
        "PMA",
        "sats_base",
        "Ventilation",
        "weight_z_score",
        "Sex",
        "ap_rate_5min",
        "GA",
        "HR_base",
    }

    right_shade_features = {
        "PMA",
        "HR_base",
        "sats_base",
        "infection",
        "weight_z_score",
        "ap_rate_5min",
        "Sex",
        "GA",
        "caffeine",
    }

    with plt.rc_context({"font.size": 8}):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_width_inches, fig_height_inches),
            squeeze=False,
        )

        plt.subplots_adjust(hspace=0.6, wspace=0.35)

        col_labels = ["A)", "B)", "C)", "D)"]

        for col_idx in range(min(4, ncols)):
            ax0 = axes[0, col_idx]
            pos = ax0.get_position()

            x_center = 0.5 * (pos.x0 + pos.x1)
            y_top = pos.y1

            fig.text(
                x_center,
                y_top + 0.01,
                col_labels[col_idx],
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        for idx in range(n_features):
            row_idx = idx // 2
            col_pair = idx % 2

            left_col = col_pair * 2
            right_col = left_col + 1

            ax_left = axes[row_idx, left_col]
            ax_right = axes[row_idx, right_col]

            (
                feat,
                xvals,
                feature_mean,
                feature_lo,
                feature_hi,
                _labels,
                kind,
                x_label,
            ) = all_feature_plots[idx]

            (
                _feat,
                centers,
                ap_results,
                subgroup_labels,
                ap_x_label,
            ) = all_apdur_plots[idx]

            if feature_mean is None or xvals is None:
                ax_left.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax_left.set_xticks([])
                ax_left.set_yticks([])
            elif kind == "cont":
                _plot_continuous_feature_panel(
                    ax=ax_left,
                    xvals=np.asarray(xvals, dtype=float),
                    mean=np.asarray(feature_mean, dtype=float),
                    lo=np.asarray(feature_lo, dtype=float),
                    hi=np.asarray(feature_hi, dtype=float),
                    x_label=x_label,
                    y_col=y_col,
                    y_lim=left_ylim,
                    shade=feat in left_shade_features,
                )
            else:
                _plot_categorical_feature_panel(
                    ax=ax_left,
                    xvals=list(xvals),
                    mean=np.asarray(feature_mean, dtype=float),
                    lo=np.asarray(feature_lo, dtype=float),
                    hi=np.asarray(feature_hi, dtype=float),
                    feature=feat,
                    x_label=x_label,
                    y_col=y_col,
                    y_lim=left_ylim,
                    categorical_point_features=categorical_point_features,
                    shade=feat in left_shade_features,
                )

            if centers is None or len(centers) == 0 or not ap_results:
                ax_right.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax_right.set_xticks([])
                ax_right.set_yticks([])
            else:
                _plot_apdur_subgroup_panel(
                    ax=ax_right,
                    centers=np.asarray(centers, dtype=float),
                    ap_results=ap_results,
                    subgroup_labels=subgroup_labels,
                    y_col=y_col,
                    y_lim=right_ylim,
                    legend_title=ap_x_label if ap_x_label is not None else "",
                    feature=feat,
                    shade=feat in right_shade_features,
                )

            remove_top_right_and_grid(ax_left)
            remove_top_right_and_grid(ax_right)

        total_slots = nrows * 2

        for slot_idx in range(total_slots):
            if slot_idx >= n_features:
                row_idx = slot_idx // 2
                col_pair = slot_idx % 2

                left_col = col_pair * 2
                right_col = left_col + 1

                axes[row_idx, left_col].axis("off")
                axes[row_idx, right_col].axis("off")

        base = out_path / f"{y_col}_{group_name}"

        fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

        plt.close(fig)

    return str(base)