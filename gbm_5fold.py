"""Run 5-fold LightGBM training using configuration stored in YAML."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%m%d")
    existing = sorted(log_dir.glob(f"output_{today}_*.log"))
    sequence = len(existing) + 1
    log_path = log_dir / f"output_{today}_{sequence:02d}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Logging to %s", log_path)
    return log_path


def prepare_features(
    frame: pd.DataFrame,
    target_column: str,
    drop_columns: list[str] | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in frame.columns:
        msg = f"Target column '{target_column}' not found in training data"
        raise KeyError(msg)

    features = frame.copy()
    drop_columns = drop_columns or []
    columns_to_drop = set(drop_columns) & set(features.columns)
    if target_column in features.columns:
        features = features.drop(columns=target_column)
    if columns_to_drop:
        features = features.drop(columns=sorted(columns_to_drop))
    target = frame[target_column]
    return features, target


def run_training(config: dict[str, Any]) -> None:
    train_path = Path(config["paths"]["train"])
    target_column = config["target"]["column"]
    drop_columns = config["target"].get("drop_columns")
    categorical = config["target"].get("categorical") or []

    logging.info("Loading training data from %s", train_path)
    data = pd.read_csv(train_path)
    features, target = prepare_features(data, target_column, drop_columns)

    cv_cfg = config.get("cv", {})
    n_splits = cv_cfg.get("n_splits", 5)
    shuffle = cv_cfg.get("shuffle", False)
    random_state = cv_cfg.get("random_state", 42)

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    params = config["model"]["params"].copy()

    train_cfg = config.get("training", {})
    num_boost_round = train_cfg.get("num_boost_round", 1000)
    early_stopping_rounds = train_cfg.get("early_stopping_rounds")
    eval_period = train_cfg.get("eval_period", 100)

    categorical = [col for col in categorical if col in features.columns]
    logging.info("Categorical features: %s", categorical)

    oof_predictions = np.zeros(len(features), dtype=float)
    fold_scores: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(features), start=1):
        logging.info("Starting fold %d/%d", fold, n_splits)
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical)

        callbacks = [lgb.log_evaluation(period=eval_period)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=True))

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        best_iteration = model.best_iteration or num_boost_round
        logging.info("Fold %d best_iteration=%d", fold, best_iteration)

        preds = model.predict(X_valid, num_iteration=best_iteration)
        oof_predictions[valid_idx] = preds
        rmse = mean_squared_error(y_valid, preds, squared=False)
        logging.info("Fold %d RMSE: %.6f", fold, rmse)
        fold_scores.append(rmse)

    overall_rmse = mean_squared_error(target, oof_predictions, squared=False)
    logging.info("Fold RMSEs: %s", ", ".join(f"{score:.6f}" for score in fold_scores))
    logging.info("OOF RMSE: %.6f", overall_rmse)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log_dir = Path(config.get("logging", {}).get("directory", "./log"))
    setup_logging(log_dir)
    run_training(config)


if __name__ == "__main__":
    main()
