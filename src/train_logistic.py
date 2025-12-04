# src/train_logistic.py

from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
)

from data_utils import load_dataset, get_X_y

# --- NEW: project root and paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "nba_pbp_features.parquet"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "logreg_winprob.pkl"
METRICS_PATH = MODEL_DIR / "logreg_metrics.json"
# -----------------------------------


def main() -> None:
    df = load_dataset(DATA_PATH)
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=skf,
        scoring="neg_log_loss",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "test_log_loss": float(log_loss(y_test, y_proba)),
        "test_brier": float(brier_score_loss(y_test, y_proba)),
        "test_auc": float(roc_auc_score(y_test, y_proba)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "best_params": grid.best_params_,
        "cv_best_score_neg_log_loss": float(grid.best_score_),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()
