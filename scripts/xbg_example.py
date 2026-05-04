import pandas as pd

from ABD_prediction import train_xgb_on_df


def main() -> None:
    df = pd.read_csv("data/input.csv")

    num_cols = [
        # add numeric columns here
    ]

    cat_cols = [
        # add categorical columns here
    ]

    class_labels = ["No response", "Response"]

    results = train_xgb_on_df(
        df=df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        class_labels=class_labels,
        outdir="outputs/xgb_run",
        id_col="ID",
        group_div=100,
        test_size=0.2,
        random_state=37,
        n_iter=500,
        cv_splits=3,
        n_jobs=4,
        verbose=1,
        n_subsamples=10,
        max_per_id=200,
        replace=True,
    )

    print("Best parameters:")
    print(results["best_params"])

    print("\nOOF metrics:")
    print(results["oof_metrics"])

    print("\nTest metrics:")
    print(results["test_metrics"])

    print("\nEnsemble metrics:")
    print(results["ensemble_metrics"])

    print("\nTop feature importances:")
    print(results["feature_importances"].head(20))


if __name__ == "__main__":
    main()