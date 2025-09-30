import pandas as pd
import numpy as np

from features import preprocessing
from visualization import plots
from evaluation import metrics


def run_pipeline():
    """Analitik aşamaları çalıştıran pipeline."""

    # --- 1. Veri Yükleme ---
    train_df = preprocessing.train_df  # preprocessing.py içinden geliyor
    test_df = preprocessing.test_df

    X_train = preprocessing.X_train
    X_valid = preprocessing.X_valid
    y_train = preprocessing.y_train
    y_valid = preprocessing.y_valid

    print("Train/Validation set hazırlandı.")
    print("X_train shape:", X_train.shape, "X_valid shape:", X_valid.shape)

    # --- 2. Feature Distribution & Correlation ---
    plots.plot_feature_distribution(X_train, kind="hist")
    plots.plot_correlation_heatmap(X_train)

    # --- 3. PCA Scatter ---
    plots.plot_pca_scatter(X_train.select_dtypes(include=["number"]), y_train)

    # --- 4. Model Eğitimi (Dummy örnek: ortalama tahmin) ---
    # Burada gerçek model yerine basit bir baseline var
    y_pred = [y_train.mean()] * len(y_train)

    # --- 5. Residuals Plot ---
    plots.plot_residuals(y_train, y_pred)

    # --- 6. Metrics Hesaplama ---
    mae = metrics.calculate_mae(y_train, y_pred)
    r2 = metrics.calculate_r2(y_train, y_pred)
    residuals = metrics.get_residuals(y_train, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # --- 7. Dashboard Input Hazırlığı ---
    dashboard_inputs = {
        "mae": mae,
        "r2": r2,
        "residuals": residuals.tolist()
    }

    print("Dashboard Inputs:", dashboard_inputs)

if __name__ == "__main__":
    run_pipeline()
