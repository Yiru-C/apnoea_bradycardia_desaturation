from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


def aggregate_by_windows_for_centers(
    data: pd.DataFrame,
    x_col: str,
    centers: np.ndarray,
    window_width: float | np.ndarray,
    y_col: str = "response",
    id_col: str = "ID",
    min_points: int = 20,
    min_unique_ids: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate y values within windows centred at `centers`.
    """
    if data.empty:
        n_centers = len(centers)
        return (
            np.full(n_centers, np.nan),
            np.zeros(n_centers, dtype=int),
            np.zeros(n_centers, dtype=int),
        )

    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()
    ids = data[id_col].to_numpy()

    n_centers = len(centers)

    values = np.full(n_centers, np.nan, dtype=float)
    counts = np.zeros(n_centers, dtype=int)
    unique_ids = np.zeros(n_centers, dtype=int)

    if np.isscalar(window_width):
        widths = np.full(n_centers, float(window_width))
    else:
        widths = np.asarray(window_width, dtype=float)

        if widths.size != n_centers:
            fallback = float(np.nanmean(widths)) if widths.size > 0 else 0.0
            widths = np.full(n_centers, fallback)

    x_range = np.nanmax(x) - np.nanmin(x) if len(x) else 1.0

    for i, center in enumerate(centers):
        width = widths[i]

        if not np.isfinite(width) or width <= 0:
            half_width = 0.5 * max(x_range, 1.0)
        else:
            half_width = 0.5 * width

        if i < n_centers - 1:
            mask = (x >= center - half_width) & (x < center + half_width)
        else:
            mask = (x >= center - half_width) & (x <= center + half_width)

        idx = np.where(mask)[0]
        counts[i] = idx.size

        if idx.size > 0:
            unique_ids[i] = np.unique(ids[idx]).size

        if counts[i] >= min_points and unique_ids[i] >= min_unique_ids:
            with np.errstate(all="ignore"):
                if y_col == "response":
                    values[i] = 100.0 * np.nanmean(y[idx].astype(float))
                else:
                    values[i] = float(np.nanmean(y[idx].astype(float)))

    return values, counts, unique_ids


def make_window_centers(
    x_min: float,
    x_max: float,
    n_windows: int,
    overlap: float,
    explicit_width: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Build window centers and window width.

    If `explicit_width` is provided, use it.
    Otherwise, choose a width from the requested number of windows and overlap.
    """
    x_range = x_max - x_min
    overlap = max(0.0, min(float(overlap), 0.95))

    if not np.isfinite(x_range) or x_range <= 0:
        return np.array([0.5 * (x_min + x_max)]), 1.0

    if explicit_width is not None and np.isfinite(explicit_width) and explicit_width > 0:
        width = float(explicit_width)

        if width >= x_range:
            return np.array([0.5 * (x_min + x_max)]), x_range

        step = width * (1.0 - overlap)

        if step <= 0:
            return np.array([0.5 * (x_min + x_max)]), width

        start = x_min + 0.5 * width
        end = x_max - 0.5 * width

        if end < start:
            return np.array([0.5 * (x_min + x_max)]), width

        centers = np.arange(start, end + 1e-8, step)
        return centers, width

    n_windows = max(int(n_windows), 2)
    width = x_range / (1.0 + (n_windows - 1) * (1.0 - overlap))

    start = x_min + 0.5 * width
    end = x_max - 0.5 * width

    if end < start:
        centers = np.array([0.5 * (x_min + x_max)])
    else:
        centers = np.linspace(start, end, num=n_windows)

    return centers, width


def merge_sparse_windows(
    centers: np.ndarray,
    width: float,
    data: pd.DataFrame,
    x_col: str,
    id_col: str,
    min_points: int,
    min_unique_ids: int,
    max_merge_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge adjacent windows from left to right if pooled windows are sparse.

    A window is sparse if it has:
    - fewer than `min_points` rows, or
    - fewer than `min_unique_ids` unique IDs.

    Merging is capped at `max_merge_size` original windows.
    """
    n_windows = len(centers)

    if n_windows == 0:
        return centers, np.array([])

    half_width = 0.5 * width

    pooled_x = data[x_col].to_numpy()
    pooled_ids = data[id_col].to_numpy()

    id_sets = []
    counts = np.zeros(n_windows, dtype=int)

    for i, center in enumerate(centers):
        if i < n_windows - 1:
            mask = (pooled_x >= center - half_width) & (pooled_x < center + half_width)
        else:
            mask = (pooled_x >= center - half_width) & (pooled_x <= center + half_width)

        idx = np.where(mask)[0]
        counts[i] = idx.size
        id_sets.append(set(pooled_ids[idx].tolist()))

    merged_groups = []
    i = 0

    while i < n_windows:
        current_indices = [i]
        current_count = counts[i]
        current_idset = set(id_sets[i])

        j = i

        while (
            (current_count < min_points or len(current_idset) < min_unique_ids)
            and j < n_windows - 1
            and (j - i + 1) < max_merge_size
        ):
            j += 1
            current_indices.append(j)
            current_count += counts[j]
            current_idset |= id_sets[j]

        merged_groups.append(current_indices)
        i = j + 1

    new_centers = []
    new_widths = []

    for group in merged_groups:
        left_idx = group[0]
        right_idx = group[-1]

        left_edge = centers[left_idx] - half_width
        right_edge = centers[right_idx] + half_width

        new_center = 0.5 * (left_edge + right_edge)
        new_width = right_edge - left_edge

        if new_width <= 0 or not np.isfinite(new_width):
            new_width = width

        new_centers.append(new_center)
        new_widths.append(new_width)

    return np.asarray(new_centers, dtype=float), np.asarray(new_widths, dtype=float)


def compute_apdur_split_curves(
    replicates: List[pd.DataFrame],
    group_fns: List[Callable[[pd.DataFrame], pd.DataFrame]],
    n_windows_apdur: int = 40,
    window_overlap_apdur: float = 0.5,
    window_width_apdur: Optional[float] = None,
    id_col: str = "ID",
    y_col: str = "response",
    min_points: int = 20,
    min_unique_ids: int = 3,
    max_merge_size: int = 10,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Compute apnoea-duration curves for one or more subgroup functions.
    """
    if not replicates:
        return np.array([]), [
            (np.array([]), np.array([]), np.array([])) for _ in group_fns
        ]

    merged = pd.concat(replicates, ignore_index=True)
    x_col = "ap_dur"

    if x_col not in merged.columns or merged.empty:
        return np.array([]), [
            (np.array([]), np.array([]), np.array([])) for _ in group_fns
        ]

    x_min = float(merged[x_col].min())
    x_max = float(merged[x_col].max())

    centers, width = make_window_centers(
        x_min=x_min,
        x_max=x_max,
        n_windows=n_windows_apdur,
        overlap=window_overlap_apdur,
        explicit_width=window_width_apdur,
    )

    centers, widths = merge_sparse_windows(
        centers=centers,
        width=width,
        data=merged,
        x_col=x_col,
        id_col=id_col,
        min_points=min_points,
        min_unique_ids=min_unique_ids,
        max_merge_size=max_merge_size,
    )

    n_new = len(centers)

    matrices = []

    for subset_fn in group_fns:
        per_rep = []

        for rep in replicates:
            subset = subset_fn(rep)

            if subset.empty:
                per_rep.append(np.full(n_new, np.nan))
            else:
                values, _, _ = aggregate_by_windows_for_centers(
                    data=subset,
                    x_col=x_col,
                    centers=centers,
                    window_width=widths,
                    y_col=y_col,
                    id_col=id_col,
                    min_points=min_points,
                    min_unique_ids=min_unique_ids,
                )
                per_rep.append(values)

        matrices.append(np.vstack(per_rep) if per_rep else np.full((1, n_new), np.nan))

    results = []

    for matrix in matrices:
        mean = np.nanmean(matrix, axis=0)
        lo = np.nanpercentile(matrix, 2.5, axis=0)
        hi = np.nanpercentile(matrix, 97.5, axis=0)
        results.append((mean, lo, hi))

    return centers, results