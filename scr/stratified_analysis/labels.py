from __future__ import annotations

import matplotlib.pyplot as plt


def y_label_from_col(y_col_name: str) -> str:
    """
    Map outcome column names to plot y-axis labels.
    
    Note: by using datasets in other formats with different column names, this function needs to be changed.
    """
    key = (y_col_name or "").lower()

    if key == "response":
        return "Apnoea with B/D (%)"

    if "hr" in key:
        return "Change in heart rate (%)"

    if "sat" in key or "sats" in key:
        return "Minimum oxygen saturation (%)"

    return f"Mean {y_col_name}"


def remove_top_right_and_grid(ax: plt.Axes) -> None:
    """Remove top/right spines and disable grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)