from typing import List

import numpy as np
import pandas as pd


def undersample_per_id(
    df: pd.DataFrame,
    id_col: str = "ID",
    max_per_id: int = 200,
    n_replicates: int = 20,
    replace: bool = True,
    random_state: int = 37,
) -> List[pd.DataFrame]:
    """
    Create undersampled replicates by capping the number of rows per ID.

    For each ID:
    - keep all rows if len(group) <= max_per_id
    - otherwise sample max_per_id rows
    """
    rng = np.random.default_rng(random_state)
    groups = list(df.groupby(id_col, observed=True))

    replicates: list[pd.DataFrame] = []

    for _ in range(n_replicates):
        parts = []

        for _, sub in groups:
            n = len(sub)

            if n <= max_per_id:
                parts.append(sub.copy())
            else:
                selected_idx = rng.choice(
                    n,
                    size=max_per_id,
                    replace=replace,
                )
                parts.append(
                    sub.iloc[selected_idx]
                    .copy()
                    .reset_index(drop=True)
                )

        rep_df = pd.concat(parts, ignore_index=True)

        shuffle_seed = int(rng.integers(1, 2**31 - 1))
        rep_df = (
            rep_df.sample(frac=1.0, random_state=shuffle_seed)
            .reset_index(drop=True)
        )

        replicates.append(rep_df)

    return replicates