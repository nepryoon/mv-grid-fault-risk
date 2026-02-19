from __future__ import annotations
import os
from typing import Any, Dict
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dictionary for one scoring request.")


class PredictResponse(BaseModel):
    fault_probability_30d: float
    risk_band: str
    model_source: str


def to_risk_band(p: float) -> str:
    if p >= 0.80: return "critical"
    if p >= 0.50: return "high"
    if p >= 0.25: return "watch"
    return "low"


def rule_based_score(f: Dict[str, Any]) -> float:
    """
    Deterministic risk score from operational features.
    Replaces the ML model for demo purposes to avoid training-data mismatch.
    Each factor contributes additively; final score is clipped to [0, 1].
    """
    score = 0.0

    # Asset age — older assets are riskier
    age = float(f.get("asset_age_years", 10) or 10)
    score += min(age / 40.0, 1.0) * 0.20

    # Recent fault history — strongest signal
    f30 = float(f.get("faults_last_30d", 0) or 0)
    f90 = float(f.get("faults_last_90d", 0) or 0)
    f180 = float(f.get("faults_last_180d", 0) or 0)
    score += min(f30 / 3.0, 1.0) * 0.30
    score += min(f90 / 8.0, 1.0) * 0.15
    score += min(f180 / 15.0, 1.0) * 0.10

    # Days since last fault — recent fault = higher risk of recurrence
    days = float(f.get("days_since_last_fault", 365) or 365)
    recency = max(0.0, 1.0 - days / 180.0)
    score += recency * 0.15

    # Asset type risk multiplier
    type_risk = {"overhead": 1.20, "underground": 0.85, "substation": 1.10, "unknown": 1.0}
    multiplier = type_risk.get(str(f.get("asset_type", "unknown")).lower(), 1.0)
    score *= multiplier

    # Region modifier
    region_mod = {"north": 1.05, "south": 0.95, "centre": 1.0, "unknown": 1.0}
    score *= region_mod.get(str(f.get("region", "unknown")).lower(), 1.0)

    # Voltage — higher voltage = higher impact, slight risk increase
    kv = float(f.get("voltage_kv", 15) or 15)
    score += min(kv / 150.0, 1.0) * 0.10

    return float(np.clip(score, 0.0, 0.99))


def create_app() -> FastAPI:
    app = FastAPI(title="MV Grid Fault Risk API", version="1.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "model_source": "rule-based-v1.1"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        p = rule_based_score(req.features)
        return PredictResponse(
            fault_probability_30d=round(p, 4),
            risk_band=to_risk_band(p),
            model_source="rule-based-v1.1",
        )

    return app


app = create_app()
