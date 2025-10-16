"""Train a LightGBM regressor for the Hull Tactical Market Prediction dataset."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Argument parsing


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
        "--date-column",
        type=str,
        default="date_id",
        help="Column that represents the chronological order of the rows.",
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
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of the chronological span reserved for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the model initialization.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (eta) for LightGBM.",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=64,
        help="Maximum number of leaves per tree.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=2000,
        help="Number of boosting iterations.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=200,
        help="Early stopping patience when monitoring validation RMSE.",
    )
    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=20,
        help="Minimum data points in one leaf.",
    )
    parser.add_argument(
        "--feature-fraction",
        type=float,
        default=0.8,
        help="Feature subsampling ratio for each tree.",
    )
    parser.add_argument(
        "--bagging-fraction",
        type=float,
        default=0.8,
        help="Row subsampling ratio for each tree.",
    )
    parser.add_argument(
        "--bagging-freq",
        type=int,
        default=1,
        help="Frequency for bagging (0 disables bagging).",
    )
    parser.add_argument(
        "--categorical",
        nargs="*",
        default=None,
        help=(
            "Column names treated as categorical by LightGBM. "
            "The columns must be present in both train and test data."
        ),
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=None,
        help="Columns to drop from both train and test before training.",
    )
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Data utilities


def read_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the competition's train and test data."""
    train = pd.read_csv(train_path, na_values=["", "NA", "NaN"])
    test = pd.read_csv(test_path, na_values=["", "NA", "NaN"])
    return train, test


def select_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    date_column: str,
    drop_columns: Iterable[str] | None,
) -> tuple[list[str], str]:
    """Select mutually available feature columns.

    The first column of the test set is treated as the row identifier in the
    submission. Additional columns provided via ``drop_columns`` are removed from
    the feature list if present. The function returns both the feature list and
    the inferred row identifier column name.
    """

    row_id_column = test.columns[0]
    invalid = {target_column, row_id_column, date_column}
    if drop_columns:
        invalid.update(drop_columns)

    common_columns = [col for col in test.columns if col in train.columns]
    features = [col for col in common_columns if col not in invalid]

    if not features:
        raise ValueError("No usable feature columns found between train and test data")

    return features, row_id_column


@dataclass
class PreparedData:
    """Container for aligned train/validation/test matrices."""

    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    row_id_column: str
    date_series: pd.Series


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
    date_column: str,
    drop_columns: Iterable[str] | None,
) -> PreparedData:
    """Prepare aligned feature matrices for training and inference."""

    if target_column not in train.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    if date_column not in train.columns:
        raise ValueError(f"Date column '{date_column}' not found in training data")

    features, row_id_column = select_features(
        train,
        test,
        target_column=target_column,
        date_column=date_column,
        drop_columns=drop_columns,
    )

    # Ensure consistent column ordering between train and test sets.
    X = train[features].apply(pd.to_numeric, errors="coerce")
    X_test = test[features].apply(pd.to_numeric, errors="coerce")

    y = pd.to_numeric(train[target_column], errors="coerce")
    date_series = pd.to_numeric(train[date_column], errors="coerce")

    return PreparedData(
        X=X,
        y=y,
        X_test=X_test,
        row_id_column=row_id_column,
        date_series=date_series,
    )


# ---------------------------------------------------------------------------
# Training utilities


def chronological_split(
    date_series: pd.Series, validation_ratio: float
) -> tuple[pd.Index, pd.Index]:
    """Generate index splits that respect chronological order."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")

    order = date_series.sort_values().index
    split_point = int(len(order) * (1 - validation_ratio))
    if split_point <= 0 or split_point >= len(order):
        raise ValueError("validation_ratio results in an empty train or validation set")

    train_indices = order[:split_point]
    valid_indices = order[split_point:]
    return train_indices, valid_indices


def train_model(
    prepared: PreparedData,
    *,
    validation_ratio: float,
    random_state: int,
    learning_rate: float,
    num_leaves: int,
    n_estimators: int,
    early_stopping_rounds: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    categorical_columns: Sequence[str] | None,
) -> tuple[lgb.LGBMRegressor, float]:
    """Train a LightGBM regressor and return the fitted model and validation RMSE."""

    train_idx, valid_idx = chronological_split(prepared.date_series, validation_ratio)

    X_train = prepared.X.loc[train_idx]
    y_train = prepared.y.loc[train_idx]
    X_valid = prepared.X.loc[valid_idx]
    y_valid = prepared.y.loc[valid_idx]

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        subsample=bagging_fraction,
        subsample_freq=bagging_freq,
        colsample_bytree=feature_fraction,
        random_state=random_state,
        n_jobs=-1,
    )

    fit_params = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_valid, y_valid)],
        "eval_metric": "rmse",
        "verbose": False,
    }

    callbacks: list[lgb.callback.Callback] = []
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    if callbacks:
        fit_params["callbacks"] = callbacks

    if categorical_columns:
        fit_params["categorical_feature"] = [
            col for col in categorical_columns if col in prepared.X.columns
        ]

    model.fit(**fit_params)

    valid_predictions = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, valid_predictions, squared=False)
    return model, rmse


# ---------------------------------------------------------------------------
# Output handling


def save_predictions(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    row_id_column: str,
    prediction_column: str,
    output_path: Path,
) -> None:
    """Persist predictions to disk in Kaggle submission format."""

    submission = pd.DataFrame({row_id_column: test[row_id_column], prediction_column: predictions})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Entry point


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(args)
    train_df, test_df = read_data(parsed.train, parsed.test)

    prepared = prepare_features(
        train_df,
        test_df,
        target_column=parsed.target,
        date_column=parsed.date_column,
        drop_columns=parsed.drop,
    )

    model, validation_rmse = train_model(
        prepared,
        validation_ratio=parsed.validation_ratio,
        random_state=parsed.random_state,
        learning_rate=parsed.learning_rate,
        num_leaves=parsed.num_leaves,
        n_estimators=parsed.n_estimators,
        early_stopping_rounds=parsed.early_stopping_rounds,
        min_data_in_leaf=parsed.min_data_in_leaf,
        feature_fraction=parsed.feature_fraction,
        bagging_fraction=parsed.bagging_fraction,
        bagging_freq=parsed.bagging_freq,
        categorical_columns=parsed.categorical,
    )
    print(f"Validation RMSE: {validation_rmse:.6f}")

    predictions = model.predict(prepared.X_test)
    save_predictions(
        test_df,
        predictions,
        row_id_column=prepared.row_id_column,
        prediction_column=parsed.prediction_column,
        output_path=parsed.output,
    )
    print(f"Predictions saved to {parsed.output.resolve()}")


if __name__ == "__main__":
    main()
