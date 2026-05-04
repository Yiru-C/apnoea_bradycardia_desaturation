import pandas as pd

from ABD_prediction.stratified_analysis import stratified_analysis


def main() -> None:
    df = pd.read_csv("data/input.csv")

    rows = [
        # Example row format expected
        # (
        #     "cont",
        #     {
        #         "x_col": "PMA",
        #         "x_label": "Postmenstrual age",
        #         "thr": [33],
        #         "labels": ["<33", "≥33"],
        #     },
        #     None,
        # ),
    ]

    results = stratified_analysis(
        df=df,
        rows=rows,
        out_path="outputs/stratified_analysis",
        id_col="ID",
        max_per_id=200,
        n_boot=50,
        replace=True,
        random_state=37,
        n_windows_apdur=80,
        window_overlap_apdur=0.5,
        window_width_apdur=None,
        n_windows_feat=40,
        window_overlap_feat=0.5,
        min_points=20,
        min_unique_ids=3,
        max_merge_size=10,
        mixed_fixed_terms=[
            "PMA",
            "GA",
            "weight_z_score",
            "HR_base",
            "sats_base",
        ],
        lme=True,
        ys=["response", "HR_change", "sats_min"],
    )

    print(results)


if __name__ == "__main__":
    main()