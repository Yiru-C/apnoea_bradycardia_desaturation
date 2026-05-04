from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sampling import undersample_per_id_bootstrap
from stratified_analysis.models import (
    fit_mixed_models_on_bootstraps,
)
from stratified_analysis.plots import (
    save_big_figure_for_group,
    save_three_apdur_plot,
)


def stable_seed_from_key(base_seed: int, key: str) -> int:
    """Create a reproducible seed from a base seed and subgroup key."""
    key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return int(base_seed + (key_hash % 10000))


def stratified_analysis(
    df: pd.DataFrame,
    rows,
    out_path: str | Path,
    id_col: str = "ID",
    max_per_id: int = 200,
    n_boot: int = 50,
    replace: bool = True,
    random_state: int = 37,
    n_windows_apdur: int = 80,
    window_overlap_apdur: float = 0.5,
    window_width_apdur: Optional[float] = None,
    n_windows_feat: int = 40,
    window_overlap_feat: float = 0.5,
    min_points: int = 20,
    min_unique_ids: int = 3,
    max_merge_size: int = 10,
    figsize_colwidth: int = 4,
    mixed_fixed_terms: Optional[List[str]] = None,
    lme: bool = True,
    reml: bool = False,
    verbose_mixed: bool = False,
    ys: List[str] = ["response"],
    alpha: float = 0.05,
    group_a: List[str] = None,
    group_b: List[str] = None,
) -> Dict[str, Any]:
    """
    Run stratified bootstrap analysis.

    Produces:
    - 3-panel apnoea-duration plots
    - grouped feature/apnoea-duration figures
    - optional mixed-model summaries saved as CSV
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    mixed_fixed_terms = mixed_fixed_terms or []

    subsets = {
        "all": df.copy(),
    }

    saved_plots: Dict[str, Dict[str, str]] = {}
    mixed_results: Dict[str, Any] = {}

    for key, sub in subsets.items():
        saved_plots[key] = {}
        mixed_results[key] = None

        if sub.empty:
            continue

        seed = stable_seed_from_key(random_state, key)

        replicates = undersample_per_id_bootstrap(
            df=sub,
            id_col=id_col,
            max_per_id=max_per_id,
            n_boot=n_boot,
            replace=replace,
            random_state=seed,
        )

        combined_base = save_three_apdur_plot(
            replicates=replicates,
            ys=ys,
            out_path=out_path,
            id_col=id_col,
            n_windows_apdur=n_windows_apdur,
            window_overlap_apdur=window_overlap_apdur,
            window_width_apdur=window_width_apdur,
            min_points=min_points,
            min_unique_ids=min_unique_ids,
            max_merge_size=max_merge_size,
            figsize=(6, 2.3),
            dpi=600,
        )

        if combined_base:
            saved_plots[key]["apdur_all_3panel_png"] = f"{combined_base}.png"
            saved_plots[key]["apdur_all_3panel_pdf"] = f"{combined_base}.pdf"

        for yplot in ys:
            base_a = save_big_figure_for_group(
                replicates=replicates,
                rows=rows,
                feature_list=group_a,
                y_col=yplot,
                out_path=out_path,
                group_name="groupA",
                id_col=id_col,
                n_windows_apdur=n_windows_apdur,
                window_overlap_apdur=window_overlap_apdur,
                n_windows_feat=n_windows_feat,
                window_overlap_feat=window_overlap_feat,
                min_points=min_points,
                min_unique_ids=min_unique_ids,
                max_merge_size=max_merge_size,
                figsize_per_panel=(figsize_colwidth * 1.0, 2.7),
                dpi=600,
            )

            if base_a:
                saved_plots[key][f"{yplot}_groupA_png"] = f"{base_a}.png"
                saved_plots[key][f"{yplot}_groupA_pdf"] = f"{base_a}.pdf"

            base_b = save_big_figure_for_group(
                replicates=replicates,
                rows=rows,
                feature_list=group_b,
                y_col=yplot,
                out_path=out_path,
                group_name="groupB",
                id_col=id_col,
                n_windows_apdur=n_windows_apdur,
                window_overlap_apdur=window_overlap_apdur,
                n_windows_feat=n_windows_feat,
                window_overlap_feat=window_overlap_feat,
                min_points=min_points,
                min_unique_ids=min_unique_ids,
                max_merge_size=max_merge_size,
                figsize_per_panel=(figsize_colwidth * 1.0, 2.7),
                dpi=600,
            )

            if base_b:
                saved_plots[key][f"{yplot}_groupB_png"] = f"{base_b}.png"
                saved_plots[key][f"{yplot}_groupB_pdf"] = f"{base_b}.pdf"

        if lme:
            mixed_results[key] = {}

            merged = pd.concat(replicates, ignore_index=True)

            for y in ys:
                rows_summaries = []

                for feature in mixed_fixed_terms:
                    if feature not in merged.columns:
                        continue

                    summary_df = fit_mixed_models_on_bootstraps(
                        replicates=replicates,
                        fixed_terms=[feature],
                        y=y,
                        random_effect=id_col,
                        alpha=alpha,
                        reml=reml,
                        verbose=verbose_mixed,
                    )

                    if summary_df is None or summary_df.empty:
                        continue

                    summary_df["feature"] = feature

                    if "term" not in summary_df.columns:
                        summary_df = (
                            summary_df.reset_index()
                            .rename(columns={"index": "term"})
                        )

                    ordered_cols = ["feature", "term"] + [
                        col
                        for col in summary_df.columns
                        if col not in ("feature", "term")
                    ]

                    rows_summaries.append(summary_df[ordered_cols])

                if not rows_summaries:
                    mixed_results[key][y] = None
                    continue

                stacked = pd.concat(
                    rows_summaries,
                    ignore_index=True,
                    sort=False,
                )

                summary_path = out_path / f"mixed_summary_{key}_{y}.csv"
                stacked.to_csv(summary_path, index=False)

                mixed_results[key][y] = str(summary_path)

    return {
        "saved_plots": saved_plots,
        "lmes": mixed_results,
    }