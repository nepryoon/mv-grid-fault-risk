from __future__ import annotations

import pandas as pd


def build_training_table(
    faults: pd.DataFrame,
    assets: pd.DataFrame,
    *,
    asset_id_col: str = "asset_id",
    event_date_col: str = "event_date",
) -> pd.DataFrame:
    """
    Build a modelling table (one row per asset per reference date).

    This is a minimal baseline intended for a portfolio demo.
    Replace/extend with your domain-grade features (load, weather proxies, maintenance history, etc.).
    """
    faults = faults.copy()
    assets = assets.copy()

    faults[event_date_col] = pd.to_datetime(faults[event_date_col], errors="coerce")
    assets["install_date"] = pd.to_datetime(assets.get("install_date"), errors="coerce")

    faults = faults.dropna(subset=[asset_id_col, event_date_col])
    assets = assets.dropna(subset=[asset_id_col])

    # Reference dates: use fault timestamps (simple baseline)
    ref = (
        faults[[asset_id_col, event_date_col]]
        .rename(columns={event_date_col: "ref_date"})
        .drop_duplicates()
        .sort_values("ref_date")
    )

    faults_sorted = faults.sort_values([asset_id_col, event_date_col])
    rows: list[dict] = []

    for _, r in ref.iterrows():
        aid = r[asset_id_col]
        ref_date = r["ref_date"]

        hist = faults_sorted[(faults_sorted[asset_id_col] == aid) & (faults_sorted[event_date_col] < ref_date)]

        def count_window(days: int) -> int:
            start = ref_date - pd.Timedelta(days=days)
            return int((hist[event_date_col] >= start).sum())

        last_fault_days = 9999
        if len(hist) > 0:
            last_fault_days = int((ref_date - hist[event_date_col].max()).days)

        rows.append(
            {
                asset_id_col: aid,
                "ref_date": ref_date,
                "faults_last_30d": count_window(30),
                "faults_last_90d": count_window(90),
                "faults_last_180d": count_window(180),
                "days_since_last_fault": last_fault_days,
            }
        )

    feat = pd.DataFrame(rows)

    # Asset metadata join
    feat = feat.merge(assets, on=asset_id_col, how="left")

    # Asset age
    feat["asset_age_years"] = (feat["ref_date"] - feat["install_date"]).dt.days / 365.25
    feat["asset_age_years"] = feat["asset_age_years"].fillna(feat["asset_age_years"].median())

    # Drop datetime columns other than ref_date to avoid dtype promotion issues in sklearn pipelines.
    datetime_cols = [
        c for c in feat.columns
        if c != "ref_date" and pd.api.types.is_datetime64_any_dtype(feat[c])
    ]
    if datetime_cols:
        feat = feat.drop(columns=datetime_cols)


    # Keep simple categoricals for encoding in training
    for col in ["asset_type", "region"]:
        if col in feat.columns:
            feat[col] = feat[col].fillna("unknown").astype(str)

    return feat


def add_target_label(
    training_table: pd.DataFrame,
    faults: pd.DataFrame,
    *,
    asset_id_col: str = "asset_id",
    event_date_col: str = "event_date",
    horizon_days: int = 30,
    target_col: str = "fault_within_30d",
) -> pd.DataFrame:
    """
    Create a binary label: whether a fault occurs within horizon_days after ref_date.
    """
    faults = faults.copy()
    faults[event_date_col] = pd.to_datetime(faults[event_date_col], errors="coerce")

    df = training_table.copy()
    df[target_col] = 0

    for idx, row in df.iterrows():
        aid = row[asset_id_col]
        ref_date = row["ref_date"]
        end_date = ref_date + pd.Timedelta(days=horizon_days)

        future = faults[
            (faults[asset_id_col] == aid)
            & (faults[event_date_col] > ref_date)
            & (faults[event_date_col] <= end_date)
        ]
        df.at[idx, target_col] = 1 if len(future) > 0 else 0

    return df
