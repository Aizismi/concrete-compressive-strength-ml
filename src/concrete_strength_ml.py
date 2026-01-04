"""
Concrete Compressive Strength Prediction (Tabular ML)
Models:
  - Ridge Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor

Dataset:
  Kaggle: Concrete Compressive Strength Data Set
  https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set

How to run:
  1) Download dataset from Kaggle and unzip locally
  2) Put the CSV path below or pass it via CLI:
       python concrete_strength_ml.py --data "path/to/Concrete_Data.csv"

Outputs:
  - Printed metrics (MAE, RMSE, R2)
  - CV metrics summary
  - Saved artifacts in ./outputs/
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

import joblib

# Optional plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_outputs_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaggle versions sometimes have minor whitespace differences.
    We'll normalize column names to be safe.
    """
    df = df.copy()
    df.columns = [c.strip().replace("  ", " ") for c in df.columns]
    return df


def find_target_column(df: pd.DataFrame) -> str:
    """
    Typical target column names seen for this dataset:
      - 'Concrete compressive strength(MPa, megapascals) '
      - 'Concrete compressive strength(MPa, megapascals)'
      - sometimes 'strength' in other variants
    We'll search robustly.
    """
    candidates = [c for c in df.columns if "compressive strength" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # pick the shortest / most likely clean one
        return sorted(candidates, key=len)[0]

    # fallback
    for c in df.columns:
        if c.lower().strip() in ["strength", "csmpa", "target", "y"]:
            return c

    raise ValueError(
        "Could not automatically find the target column. "
        "Please edit find_target_column() or rename your CSV column."
    )


def regression_report(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": label,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def build_preprocessor(X: pd.DataFrame):
    """
    For this dataset, everything should be numeric.
    Still, we build a robust numeric pipeline with imputation + scaling.
    """
    numeric_features = X.columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop"
    )

    return preprocessor


def plot_predictions(y_true, y_pred, title, out_path):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.title(title)
    # line y=x
    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=False, default="Concrete_Data.csv",
                        help="Path to the CSV file (downloaded locally from Kaggle).")
    parser.add_argument("--outdir", type=str, required=False, default="outputs",
                        help="Directory to save models and plots.")
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--test_size", type=float, required=False, default=0.2)
    parser.add_argument("--cv_folds", type=int, required=False, default=5)
    args = parser.parse_args()

    ensure_outputs_dir(args.outdir)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"CSV not found at: {args.data}\n\n"
            "Make sure you've downloaded/unzipped the Kaggle dataset locally and "
            "provide the correct path, e.g.\n"
            '  python concrete_strength_ml.py --data "C:\\path\\to\\Concrete_Data.csv"\n'
        )

    df = pd.read_csv(args.data)
    # Clean column names (removes trailing/leading spaces)
    df.columns = df.columns.str.strip()

    # Explicit target column for this dataset
    target_col = "concrete_compressive_strength"
    
    df = standardize_column_names(df)

    # Identify target column and split X/y
    #target_col = find_target_column(df)
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    preprocessor = build_preprocessor(X_train)

    # -----------------------------
    # Model definitions
    # -----------------------------
    ridge = Ridge(alpha=1.0, random_state=args.seed)

    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=args.seed,
        n_jobs=-1,
        max_depth=None
    )

    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=args.seed,
        n_jobs=-1
    )

    models = {
        "Ridge": ridge,
        "RandomForest": rf,
        "XGBoost": xgb
    }

    # -----------------------------
    # Cross-validation setup
    # -----------------------------
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    all_results = []

    print("\n==============================")
    print("DATASET SUMMARY")
    print("==============================")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Target: {target_col}")
    print("\nFeature columns:")
    for c in X.columns:
        print(" -", c)

    print("\n==============================")
    print("CROSS-VALIDATION (Train set)")
    print("==============================")

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        cv_scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        mae = -np.mean(cv_scores["test_mae"])
        rmse = -np.mean(cv_scores["test_rmse"])
        r2 = np.mean(cv_scores["test_r2"])

        print(f"\n{name}:")
        print(f"  CV MAE : {mae:.4f}")
        print(f"  CV RMSE: {rmse:.4f}")
        print(f"  CV R2  : {r2:.4f}")

        all_results.append({
            "Model": name,
            "CV_MAE": mae,
            "CV_RMSE": rmse,
            "CV_R2": r2
        })

    results_df = pd.DataFrame(all_results).sort_values(by="CV_RMSE", ascending=True)
    results_path = os.path.join(args.outdir, "cv_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\n==============================")
    print("CV RESULTS (Saved)")
    print("==============================")
    print(results_df)
    print(f"\nSaved to: {results_path}")

    # Pick best by CV_RMSE
    best_model_name = results_df.iloc[0]["Model"]
    best_estimator = models[best_model_name]

    # Fit best model on full train set, evaluate on hold-out test
    print("\n==============================")
    print(f"TRAIN BEST MODEL + TEST EVALUATION: {best_model_name}")
    print("==============================")

    best_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", best_estimator)
    ])

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    test_metrics = regression_report(y_test, y_pred, label=best_model_name)
    print(f"Test MAE : {test_metrics['MAE']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test R2  : {test_metrics['R2']:.4f}")

    # Save model
    model_path = os.path.join(args.outdir, f"best_model_{best_model_name}.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\nSaved trained pipeline to: {model_path}")

    # Save test predictions
    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    })
    pred_path = os.path.join(args.outdir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path}")

    # Plot predicted vs actual
    plot_path = os.path.join(args.outdir, f"pred_vs_actual_{best_model_name}.png")
    plot_predictions(y_test.values, y_pred, f"Predicted vs Actual ({best_model_name})", plot_path)
    print(f"Saved plot to: {plot_path}")

    # Feature importance (tree models only)
    # Ridge: coefficients after preprocessing are not directly interpretable without extra steps,
    # so we focus on RF/XGB feature importances if best model is tree-based.
    if best_model_name in ["RandomForest", "XGBoost"]:
        model = best_pipe.named_steps["model"]

        # For ColumnTransformer, numeric features remain in same order
        feature_names = X.columns.tolist()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi = fi.sort_values(by="importance", ascending=False)

            fi_path = os.path.join(args.outdir, f"feature_importance_{best_model_name}.csv")
            fi.to_csv(fi_path, index=False)
            print(f"\nSaved feature importance to: {fi_path}")

            # Plot top 10
            top = fi.head(10).iloc[::-1]  # reverse for horizontal plot
            plt.figure()
            plt.barh(top["feature"], top["importance"])
            plt.xlabel("Importance")
            plt.title(f"Top 10 Feature Importances ({best_model_name})")
            plt.tight_layout()
            fi_plot_path = os.path.join(args.outdir, f"feature_importance_{best_model_name}.png")
            plt.savefig(fi_plot_path, dpi=200)
            plt.close()
            print(f"Saved feature importance plot to: {fi_plot_path}")

    print("\nDONE.\n")
    


if __name__ == "__main__":
    main()
