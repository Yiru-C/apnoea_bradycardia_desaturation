import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def group_stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 37,
    response_col: str = "response",
    pma_col: str = "PMA",
    ap_dur_col: str = "ap_dur",
    group_id_col = "group_ID",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-aware stratified split.

    Stratification uses:
    - PMA bin
    - response
    - apnoea duration bin

    Rare strata are handled separately to avoid sklearn split errors.
    """

    group_df = df.groupby(group_id_col, observed=True).first().reset_index()

    group_df["PMA_bin"] = pd.cut(
        group_df[pma_col],
        bins=[-np.inf, 33, np.inf],
        labels=False,
    )

    group_df["ap_dur_bin"] = pd.cut(
        group_df[ap_dur_col],
        bins=[-np.inf, 10, np.inf],
        labels=False,
    )

    group_df["stratify_col"] = (
        group_df["PMA_bin"].astype(str)
        + "_"
        + group_df[response_col].astype(str)
        + "_"
        + group_df["ap_dur_bin"].astype(str)
    )

    strat_counts = group_df["stratify_col"].value_counts()
    rare_mask = group_df["stratify_col"].isin(
        strat_counts[strat_counts < 2].index
    )

    common = group_df[~rare_mask].reset_index(drop=True)
    rare = group_df[rare_mask].reset_index(drop=True)

    rng = np.random.default_rng(random_state)

    train_common = group_df.iloc[0:0]
    test_common = group_df.iloc[0:0]

    if len(common) >= 2:
        try:
            train_common, test_common = train_test_split(
                common,
                test_size=test_size,
                random_state=random_state,
                stratify=common["stratify_col"],
            )
        except ValueError:
            train_common, test_common = train_test_split(
                common,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )
    else:
        train_common = common

    if len(rare) >= 2:
        rare_train, rare_test = train_test_split(
            rare,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
    elif len(rare) == 1:
        rare_group = rare.iloc[[0]]
        if rng.random() < float(test_size):
            rare_train = rare.iloc[0:0]
            rare_test = rare_group
        else:
            rare_train = rare_group
            rare_test = rare.iloc[0:0]
    else:
        rare_train = rare.iloc[0:0]
        rare_test = rare.iloc[0:0]

    train_ids = pd.concat([train_common, rare_train])[group_id_col]
    test_ids = pd.concat([test_common, rare_test])[group_id_col]

    if train_ids.nunique() == 0 or test_ids.nunique() == 0:
        all_groups = group_df[group_id_col].unique()
        n_groups = len(all_groups)

        if n_groups < 2:
            raise ValueError("Need at least two groups for a train/test split.")

        n_test = max(1, int(round(test_size * n_groups)))
        shuffled = list(all_groups)
        rng.shuffle(shuffled)

        test_ids = pd.Series(shuffled[:n_test])
        train_ids = pd.Series(shuffled[n_test:])

        if len(train_ids) == 0:
            train_ids = pd.Series(shuffled[:1])
            test_ids = pd.Series(shuffled[1:])

    train_df = (
        df[df[group_id_col].isin(train_ids)]
        .drop(columns=[group_id_col])
        .reset_index(drop=True)
    )

    test_df = (
        df[df[group_id_col].isin(test_ids)]
        .drop(columns=[group_id_col])
        .reset_index(drop=True)
    )

    return train_df, test_df