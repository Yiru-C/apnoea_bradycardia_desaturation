This repository contains Python code for two related analysis workflows:

1. **ABD prediction** using a group-aware XGBoost classification pipeline.
2. **Stratified apnoea-duration analysis** using undersampled aggregation, sliding-window summaries, plots, and optional mixed-model summaries.

The code is organised as an installable Python package using a `src/` layout. Example usage of the functions can be found in `scripts/`.

---

## Repository structure

```text
.
├── pyproject.toml
├── README.md
├── src/
│   ├── sampling.py
│   ├── ABD_prediction/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── splitting.py
│   │   ├── plotting.py
│   │   └── training.py
│   └── stratified_analysis/
│       ├── __init__.py
│       ├── bootstrap.py
│       ├── labels.py
│       ├── models.py
│       ├── pipeline.py
│       ├── plots.py
│       └── windows.py
└── scripts/
    ├── train_example.py
    └── run_stratified_analysis.py