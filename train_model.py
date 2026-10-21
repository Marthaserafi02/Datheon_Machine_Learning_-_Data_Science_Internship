from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "california_housing.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "median_house_value"
RANDOM_STATE = 42


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create ratio features while protecting against division by zero."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["rooms_per_household"] = (
            X["total_rooms"] / X["households"].replace(0, np.nan)
        )
        X["population_per_household"] = (
            X["population"] / X["households"].replace(0, np.nan)
        )
        X["bedrooms_per_room"] = (
            X["total_bedrooms"] / X["total_rooms"].replace(0, np.nan)
        )
        return X


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop=None)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "Model": name,
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "R2": float(r2_score(y_test, predictions)),
    }, predictions


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place california_housing.csv in data/."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' was not found.")

    print("=" * 72)
    print("CALIFORNIA HOUSING PRICE PREDICTION")
    print("=" * 72)
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False).to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Save a numeric correlation heatmap for exploratory analysis.
    numeric_for_corr = df.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_for_corr.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("California Housing: Numeric Feature Correlations")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Feature engineering is inside the pipeline so it runs consistently.
    numeric_features = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "rooms_per_household", "population_per_household", "bedrooms_per_room",
    ]
    categorical_features = ["ocean_proximity"]

    def build_pipeline(model):
        return Pipeline([
            ("feature_engineering", HousingFeatureEngineer()),
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ])

    results = []
    predictions_by_model = {}
    models = {}

    # Ordinary Least Squares baseline
    ols = build_pipeline(LinearRegression())
    ols.fit(X_train, y_train)
    metrics, predictions = evaluate_model("OLS", ols, X_test, y_test)
    results.append(metrics)
    predictions_by_model["OLS"] = predictions
    models["OLS"] = ols

    # Tune regularization strength with cross-validation on the training data only.
    alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    ridge_search = GridSearchCV(
        estimator=build_pipeline(Ridge()),
        param_grid={"model__alpha": alpha_values},
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    ridge_search.fit(X_train, y_train)
    metrics, predictions = evaluate_model("Ridge", ridge_search.best_estimator_, X_test, y_test)
    metrics["Best_Alpha"] = ridge_search.best_params_["model__alpha"]
    results.append(metrics)
    predictions_by_model["Ridge"] = predictions
    models["Ridge"] = ridge_search.best_estimator_

    lasso_search = GridSearchCV(
        estimator=build_pipeline(Lasso(max_iter=20000, tol=0.001)),
        param_grid={"model__alpha": alpha_values},
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    lasso_search.fit(X_train, y_train)
    metrics, predictions = evaluate_model("Lasso", lasso_search.best_estimator_, X_test, y_test)
    metrics["Best_Alpha"] = lasso_search.best_params_["model__alpha"]
    results.append(metrics)
    predictions_by_model["Lasso"] = predictions
    models["Lasso"] = lasso_search.best_estimator_

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nTest-set model comparison:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nBest Ridge alpha: {ridge_search.best_params_['model__alpha']}")
    print(f"Best Lasso alpha: {lasso_search.best_params_['model__alpha']}")

    # Plot actual vs predicted for all models.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, name in zip(axes, ["OLS", "Ridge", "Lasso"]):
        pred = predictions_by_model[name]
        ax.scatter(y_test, pred, alpha=0.25, s=10)
        low = min(float(y_test.min()), float(pred.min()))
        high = max(float(y_test.max()), float(pred.max()))
        ax.plot([low, high], [low, high], "r--", linewidth=1.5)
        ax.set_title(f"{name}: Actual vs Predicted")
        ax.set_xlabel("Actual median house value ($)")
        ax.set_ylabel("Predicted median house value ($)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Plot residual distributions.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, name in zip(axes, ["OLS", "Ridge", "Lasso"]):
        residuals = y_test.to_numpy() - predictions_by_model[name]
        sns.histplot(residuals, bins=45, stat="density", kde=True, color="steelblue", ax=ax)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{name}: Residual Distribution")
        ax.set_xlabel("Residual (actual - predicted), $")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residual_distributions.png", dpi=160)
    plt.close(fig)

    joblib.dump(models["Ridge"], OUTPUT_DIR / "best_ridge_model.joblib")
    joblib.dump(models["Lasso"], OUTPUT_DIR / "best_lasso_model.joblib")

    best_row = results_df.loc[results_df["RMSE"].idxmin()]
    print(f"\nLowest test RMSE: {best_row['Model']} ({best_row['RMSE']:,.2f})")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
