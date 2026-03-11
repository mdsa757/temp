# ============================================================
# main.py -- WQI Prediction System
# Full end-to-end pipeline orchestrator.
# Execution order (.clauderules):
#   1. XGBoost Baseline
#   2. LSTM
#   3. DFNN
#   4. GRU
#   5. XGBoost Tuned (Optuna, nested CV, purged augmentation)
#   6. SHAP analysis for all 5 models
#   7. Save comparison table + best model name
# ============================================================

import os
import random

import numpy as np
import pandas as pd

from src.config import (
    CFG_RANDOM_SEED,
    PATH_BEST_MODEL_TXT,
    PATH_MODEL_COMPARISON_CSV,
    PATH_MODELS_DIR,
    PATH_RESULTS_DIR,
)

# ── Reproducibility: set before any imports that use RNG ─────────────────────
random.seed(CFG_RANDOM_SEED)
np.random.seed(CFG_RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(CFG_RANDOM_SEED)

# ── Model pipelines ───────────────────────────────────────────────────────────
from src.models.wqi_xgboost_model       import run_xgboost_pipeline
from src.models.wqi_lstm_model          import run_lstm_pipeline
from src.models.wqi_dfnn_model          import run_dfnn_pipeline
from src.models.wqi_gru_model           import run_gru_pipeline
from src.models.wqi_xgboost_tuned_model import run_xgboost_tuned_pipeline

# ── SHAP analysis ─────────────────────────────────────────────────────────────
from src.shap_analysis.wqi_shap_xgboost       import run_shap_xgboost
from src.shap_analysis.wqi_shap_lstm          import run_shap_lstm
from src.shap_analysis.wqi_shap_dfnn          import run_shap_dfnn
from src.shap_analysis.wqi_shap_gru           import run_shap_gru
from src.shap_analysis.wqi_shap_xgboost_tuned import run_shap_xgboost_tuned


def main() -> None:
    os.makedirs(PATH_MODELS_DIR, exist_ok=True)
    os.makedirs(PATH_RESULTS_DIR, exist_ok=True)

    list_metrics = []

    # ── 1. XGBoost Baseline ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 1: XGBoost Baseline")
    print("=" * 60)
    list_metrics.append(run_xgboost_pipeline())

    # ── 2. LSTM ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 2: LSTM")
    print("=" * 60)
    list_metrics.append(run_lstm_pipeline())

    # ── 3. DFNN ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 3: DFNN")
    print("=" * 60)
    list_metrics.append(run_dfnn_pipeline())

    # ── 4. GRU ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 4: GRU")
    print("=" * 60)
    list_metrics.append(run_gru_pipeline())

    # ── 5. XGBoost Tuned ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 5: XGBoost Tuned (Optuna + nested CV)")
    print("=" * 60)
    list_metrics.append(run_xgboost_tuned_pipeline())

    # ── Build comparison DataFrame ────────────────────────────────────────────
    df_results = pd.DataFrame(list_metrics)
    df_results = df_results.sort_values(
        by=["R2 Score", "RMSE"],
        ascending=[False, True],
    ).reset_index(drop=True)
    df_results.insert(0, "Rank", range(1, len(df_results) + 1))

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON TABLE")
    print("=" * 60)
    print(df_results.to_string(index=False))

    # ── Save comparison CSV ───────────────────────────────────────────────────
    df_results.to_csv(PATH_MODEL_COMPARISON_CSV, index=False)
    print(f"\nComparison table saved -> {PATH_MODEL_COMPARISON_CSV}")

    # ── Find and save best model (row 0 after sort) ───────────────────────────
    str_best_model_name = str(df_results.loc[0, "Model"])
    with open(PATH_BEST_MODEL_TXT, "w") as f:
        f.write(str_best_model_name)
    print(f"Best model: {str_best_model_name}")
    print(f"Best model name saved -> {PATH_BEST_MODEL_TXT}")

    # ── 6. SHAP Analysis (all 5 models) ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)

    print("\n[SHAP] Running XGBoost Baseline...")
    run_shap_xgboost()

    print("\n[SHAP] Running LSTM...")
    run_shap_lstm()

    print("\n[SHAP] Running DFNN...")
    run_shap_dfnn()

    print("\n[SHAP] Running GRU...")
    run_shap_gru()

    print("\n[SHAP] Running XGBoost Tuned...")
    run_shap_xgboost_tuned()

    print("\n" + "=" * 60)
    print("ALL DONE. Check outputs/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
