from __future__ import annotations

from dataclasses import dataclass
import yaml


@dataclass(frozen=True)
class Settings:
    # Paths
    raw_faults_csv: str
    raw_assets_csv: str
    processed_parquet: str
    model_dir: str

    # MLflow
    tracking_uri: str
    experiment_name: str
    registered_model_name: str

    # Training
    target_column: str
    time_column: str
    horizon_days: int
    test_size_days: int

    # Project
    seed: int


def load_settings(path: str) -> Settings:
    """Load project settings from YAML (single source of truth)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return Settings(
        raw_faults_csv=cfg["paths"]["raw_faults_csv"],
        raw_assets_csv=cfg["paths"]["raw_assets_csv"],
        processed_parquet=cfg["paths"]["processed_parquet"],
        model_dir=cfg["paths"]["model_dir"],
        tracking_uri=cfg["mlflow"]["tracking_uri"],
        experiment_name=cfg["mlflow"]["experiment_name"],
        registered_model_name=cfg["mlflow"]["registered_model_name"],
        target_column=cfg["training"]["target_column"],
        time_column=cfg["training"]["time_column"],
        horizon_days=int(cfg["training"]["horizon_days"]),
        test_size_days=int(cfg["training"]["test_size_days"]),
        seed=int(cfg["project"]["seed"]),
    )
