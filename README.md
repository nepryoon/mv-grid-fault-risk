# MV Grid Fault Risk

**Live docs:** https://www.neuromorphicinference.com/demos/mv-grid-fault-risk/  
**Proof Ledger:** https://www.neuromorphicinference.com/evidence/#mv-grid-fault-risk  
**API docs (live, rule-based endpoint):** https://mv-grid-fault-risk-api.onrender.com/docs  
**Model artefact (release):** https://github.com/nepryoon/mv-grid-fault-risk/releases

Independent, open-source ML application (Apache 2.0) for scoring **medium-voltage (MV) grid fault risk** and prioritising preventive interventions. It is a portfolio project, not an Enel system and not a production system.

**Keywords:** feature engineering, time-based validation, MLflow experiment tracking, artefact versioning, FastAPI, Docker, GitHub Actions.

---

## What this system does

- Builds a **risk score** for MV assets/segments from structured operational data.
- Produces **monitoring-ready outputs** (predictable schema, stable interfaces).
- Ships a **serving layer** (API) and a **demo UI** for interactive scoring.
- Tracks experiments and artefacts for **reproducibility and rollback**.
- **Scope:** training uses a scikit-learn HistGradientBoostingClassifier on a time-based split (older data for training, the most recent window for testing) and is started manually through a GitHub Actions workflow. The live API endpoint returns a transparent rule-based score (`model_source: rule-based-v1.1`), not the trained model.

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
