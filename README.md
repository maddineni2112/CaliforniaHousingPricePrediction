# California Housing Price Prediction

This repository contains the original course-project notebook for predicting median house values from the California Housing dataset.

The notebook covers:

- exploratory data analysis and data-quality checks
- z-score and IQR outlier filtering
- feature scaling, degree-2 polynomial features, and recursive feature elimination
- a comparison of linear, tree, random-forest, gradient-boosting, XGBoost, voting, and stacking regressors
- cross-validation and hyperparameter search
- predicted-versus-actual and residual diagnostics

## Main deliverable

- `project_V1-0.ipynb` - the original end-to-end analysis notebook

The notebook loads the public dataset with `sklearn.datasets.fetch_california_housing`, so no local CSV, market feed, web application, or service credentials are required.

## Run the notebook

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook project_V1-0.ipynb
```

The target, `MedHouseVal`, is measured in units of $100,000. The results are for coursework and model comparison, not individual property appraisal.
