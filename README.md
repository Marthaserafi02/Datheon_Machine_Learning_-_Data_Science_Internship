# California Housing Price Prediction

**Datheon Machine Learning & Data Science Internship-Task 2**

## Objective
Predict California district median house values and compare Ordinary Least Squares (OLS), Ridge (L2), and Lasso (L1) regression using RMSE, MAE, and R².

## Dataset
`data/california_housing.csv` contains 20,640 rows and 10 columns from the California Housing dataset. The target is `median_house_value`. The dataset contains missing values in `total_bedrooms`, which are handled by the preprocessing pipeline.

## Methods
- Exploratory data analysis and numeric correlation heatmap
- Feature engineering: rooms per household, population per household, and bedrooms per room
- Median imputation for numeric columns and most-frequent imputation for categorical columns
- Standard scaling of numeric features and one-hot encoding of `ocean_proximity`
- Train/test split with a fixed random seed
- OLS baseline, plus Ridge and Lasso tuned with `GridSearchCV`
- Test-set evaluation using RMSE, MAE, and R²
- Actual-vs-predicted and residual-distribution plots

All imputers, scalers, and encoders are fitted inside the model pipeline on training folds only to reduce data leakage. Ridge and Lasso regularize coefficients, helping address multicollinearity. The correlation heatmap is a diagnostic; features are not removed solely because they are correlated.

## Project structure
```text
Datheon_Machine_Learning_&_Data_Science_Internship/
├── data/
│   └── california_housing.csv
├── outputs/                  # Generated when the script runs
├── src/
│   └── train_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup
Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run
From the project root:
```bash
python src/train_model.py
```

The script writes:
- `outputs/model_comparison.csv`
- `outputs/correlation_heatmap.png`
- `outputs/actual_vs_predicted.png`
- `outputs/residual_distributions.png`
- `outputs/best_ridge_model.joblib`
- `outputs/best_lasso_model.joblib`

## Metrics
- **RMSE:** penalizes larger prediction errors more strongly; lower is better.
- **MAE:** average absolute prediction error; lower is better.
- **R²:** proportion of target variance explained by the model; higher is better.

The script prints the measured test-set metrics and the best Ridge/Lasso alpha values. Report those actual results in any internship submission rather than guessing model performance.

## Notes
- The dataset is provided as a CSV file in this repository.
- House values in the source dataset are capped at $500,001, which can limit prediction accuracy for the most expensive districts.
- This is an educational prediction project, not a property valuation service.
