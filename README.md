# MV Grid Fault Risk Scoring Platform

**Live docs:** https://www.neuromorphicinference.com/demos/mv-grid-fault-risk/  
**Proof Ledger:** https://www.neuromorphicinference.com/evidence/#mv-grid-fault-risk  
**API docs (live):** https://mv-grid-fault-risk-api.onrender.com/docs  
**Model artefact (release):** https://github.com/nepryoon/mv-grid-fault-risk/releases

Production-first, end-to-end ML system for predicting **medium-voltage (MV) grid fault risk** and prioritising preventive interventions.

**Keywords:** scalable ML pipelines, CI/CD for ML, feature engineering, artefact versioning, model serving, inference, monitoring-ready outputs, automated retraining, FastAPI, Docker, MLflow.

---

## What this system does

- Builds a **risk score** for MV assets/segments from structured operational data.
- Produces **monitoring-ready outputs** (predictable schema, stable interfaces).
- Ships a **serving layer** (API) and a **demo UI** for interactive scoring.
- Tracks experiments and artefacts for **reproducibility and rollback**.

---

## Architecture (end-to-end)

```text
Raw data  →  Validation  →  Feature engineering  →  Train/Evaluate  →  Artefact registry
  |                                                    |                   |
  └───────────────>  Training table (versioned)        └── MLflow tracking  |
                                                                  |        |
                                                                  v        v
                                                           FastAPI serving → Demo UI
                                                                  |
                                                                  v
                                                         Monitoring-ready outputs
