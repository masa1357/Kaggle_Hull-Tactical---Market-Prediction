"""Train a LightGBM regressor for the Hull Tactical Market Prediction dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def parse_arguments(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("train.csv"),
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("test.csv"),
        help="Path to the test CSV file used for inference.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="forward_returns",
        help="Name of the target column in the training data.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="prediction",
        help="Name of the column that will store predictions in the output file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lgbm_predictions.csv"),
        help="Where to save the predictions CSV file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/validation split and the model.",
    )
    return parser.parse_args(args)


def read_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the competition's train and test data."""
    train = pd.read_csv(train_path, na_values=["", "NA", "NaN"])
    test = pd.read_csv(test_path, na_values=["", "NA", "NaN"])
    return train, test


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    """Prepare aligned feature matrices for training and inference."""
    if target_column not in train.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    row_id_column = test.columns[0]

    common_columns = [col for col in test.columns if col in train.columns]
    features = [col for col in common_columns if col != row_id_column and col != target_column]

    if not features:
        raise ValueError("No common feature columns found between train and test data")

    X = train[features].apply(pd.to_numeric, errors="coerce")
    y = train[target_column]
    X_test = test[features].apply(pd.to_numeric, errors="coerce")

    return X, y, X_test, row_id_column


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[lgb.LGBMRegressor, float]:
    """Train a LightGBM regressor and return the fitted model and validation RMSE."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        early_stopping_rounds=100,
        verbose=False,
    )

    valid_predictions = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, valid_predictions, squared=False)
    return model, rmse


def save_predictions(
    test: pd.DataFrame,
    predictions: np.ndarray,
    row_id_column: str,
    prediction_column: str,
    output_path: Path,
) -> None:
    """Persist predictions to disk in Kaggle submission format."""
    submission = pd.DataFrame({row_id_column: test[row_id_column], prediction_column: predictions})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(args)
    train, test = read_data(parsed.train, parsed.test)
    X, y, X_test, row_id_column = prepare_features(train, test, parsed.target)
    model, validation_rmse = train_model(X, y, parsed.test_size, parsed.random_state)
    print(f"Validation RMSE: {validation_rmse:.6f}")

    predictions = model.predict(X_test)
    save_predictions(
        test,
        predictions,
        row_id_column=row_id_column,
        prediction_column=parsed.prediction_column,
        output_path=parsed.output,
    )
    print(f"Predictions saved to {parsed.output.resolve()}")


if __name__ == "__main__":
    main()
