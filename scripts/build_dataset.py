from mvgrid.config import load_settings
from mvgrid.io import read_csv, write_parquet
from mvgrid.features import build_training_table, add_target_label


if __name__ == "__main__":
    cfg_path = "configs/config.yaml"
    s = load_settings(cfg_path)

    faults = read_csv(s.raw_faults_csv)
    assets = read_csv(s.raw_assets_csv)

    table = build_training_table(
        faults=faults,
        assets=assets,
        asset_id_col="asset_id",
        event_date_col=s.time_column,
    )

    table = add_target_label(
        training_table=table,
        faults=faults,
        asset_id_col="asset_id",
        event_date_col=s.time_column,
        horizon_days=s.horizon_days,
        target_col=s.target_column,
    )

    write_parquet(table, s.processed_parquet)
    print(f"Saved training table to: {s.processed_parquet}")
