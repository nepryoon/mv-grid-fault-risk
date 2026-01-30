from __future__ import annotations

import os
from typing import Any, Dict
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from scripts.fetch_model import ensure_model_present


class PredictRequest(BaseModel):
    """
    Keep the schema simple for demo purposes.
    In production, define explicit fields aligned to your feature store / schema registry.
    """
    features: Dict[str, Any] = Field(..., description="Feature dictionary for one scoring request.")


class PredictResponse(BaseModel):
    fault_probability_30d: float
    risk_band: str
    model_source: str


def to_risk_band(p: float) -> str:
    if p >= 0.80:
        return "critical"
    if p >= 0.50:
        return "high"
    if p >= 0.25:
        return "watch"
    return "low"


def create_app() -> FastAPI:
    app = FastAPI(title="MV Grid Fault Risk API", version="1.0.0")
    
    # Allow cross-origin requests for portfolio demos (Cloudflare site → Render API).
    # In production, restrict allow_origins to your domain(s).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure we have a local model directory (models/latest) or pull from MODEL_URL.
    model_dir = ensure_model_present()
    model = mlflow.sklearn.load_model(str(model_dir))
    # Expected input schema (captured from training if available)
    expected_cols = list(getattr(model, "feature_names_in_", []))


    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "model_dir": str(model_dir)}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        features = dict(req.features)

        # Sensible demo default: the training table included asset_id, so ensure it's present.
        features.setdefault("asset_id", "demo_asset")

        X = pd.DataFrame([features])

        # Align to the training schema if the model exposes it
        if expected_cols:
            for c in expected_cols:
                if c not in X.columns:
                    X[c] = np.nan
            X = X[expected_cols]

        p = float(model.predict_proba(X)[:, 1][0])
        return PredictResponse(
            fault_probability_30d=p,
            risk_band=to_risk_band(p),
            model_source=str(model_dir),
        )


    return app


app = create_app()
