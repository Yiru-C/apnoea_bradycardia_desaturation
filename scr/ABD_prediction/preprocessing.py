from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_onehot(drop: str = "first") -> OneHotEncoder:
    """Return OneHotEncoder compatible with different sklearn versions."""
    try:
        return OneHotEncoder(
            drop=drop,
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            drop=drop,
            handle_unknown="ignore",
            sparse=False,
        )


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", make_onehot(drop="first"), cat_cols),
        ],
        remainder="drop",
    )