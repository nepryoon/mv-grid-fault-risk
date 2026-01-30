from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mvgrid.config import Settings
from mvgrid.io import read_parquet


def time_based_split(df: pd.DataFrame, time_col: str, test_size_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time (train = older, test = most recent window).
    This is more realistic than random split for operational prediction.
    """
    df = df.sort_values(time_col)
    cutoff = df[time_col].max() - pd.Timedelta(days=test_size_days)
    train_df = df[df[time_col] <= cutoff]
    test_df = df[df[time_col] > cutoff]
    return train_df, test_df


def _safe_binary_metrics(y_true: np.ndarray, p_pred: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute robust metrics even when the test set has a single class.
    """
    metrics: dict[str, float] = {"roc_auc": 0.0, "pr_auc": 0.0, "f1": 0.0}
    if len(np.unique(y_true)) <= 1:
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(y_true, p_pred))
    metrics["pr_auc"] = float(average_precision_score(y_true, p_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred))
    return metrics


def train_model(settings: Settings, config_path: str) -> None:
    """
    Train a baseline classifier and track everything in MLflow.
    Exports the artefact to models/latest for simple serving.
    """
    # Load processed training table created in Step 3
    df = read_parquet(settings.processed_parquet).copy()

    # Basic hygiene
    df["ref_date"] = pd.to_datetime(df["ref_date"], errors="coerce")
    df = df.dropna(subset=["ref_date", settings.target_column])

    # Time split
    train_df, test_df = time_based_split(df, "ref_date", settings.test_size_days)

    y_train = train_df[settings.target_column].astype(int).to_numpy()
    y_test = test_df[settings.target_column].astype(int).to_numpy()

    # Feature matrix
    drop_cols = ["ref_date", settings.target_column]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    categorical = [c for c in X_train.columns if X_train[c].dtype == "object"]
    numeric = [c for c in X_train.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        random_state=settings.seed,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    # MLflow configuration
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    with mlflow.start_run(run_name="mv_fault_risk_train") as run:
        # Log training context
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("model_type", "HistGradientBoostingClassifier")
        mlflow.log_param("test_size_days", settings.test_size_days)
        mlflow.log_param("horizon_days", settings.horizon_days)
        mlflow.log_param("n_train_rows", int(len(train_df)))
        mlflow.log_param("n_test_rows", int(len(test_df)))
        mlflow.log_param("n_features", int(X_train.shape[1]))

        # Train
        pipe.fit(X_train, y_train)

        # Evaluate
        p_test = pipe.predict_proba(X_test)[:, 1] if len(test_df) > 0 else np.array([])
        y_pred = (p_test >= 0.5).astype(int) if len(p_test) > 0 else np.array([])

        metrics = _safe_binary_metrics(y_test, p_test, y_pred) if len(p_test) > 0 else {"roc_auc": 0.0, "pr_auc": 0.0, "f1": 0.0}
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=settings.registered_model_name,
            input_example=X_test.head(5) if len(X_test) > 0 else X_train.head(5),
        )

        # Export a local artefact for serving
        model_dir = Path(settings.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        export_path = model_dir / "latest"
        mlflow.sklearn.save_model(pipe, path=str(export_path))

        print(f"MLflow run id: {run.info.run_id}")
        print(f"Metrics: {metrics}")
        print(f"Exported model to: {export_path}")
