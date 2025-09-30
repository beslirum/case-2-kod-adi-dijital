from sklearn.metrics import mean_absolute_error, r2_score  # MAE ve R² için
import numpy as np  # Residuals için sayısal işlemler
 
__all__ = ["calculate_mae", "calculate_r2", "calculate_residuals"]

def calculate_mae(y_true, y_pred):
    """Gerçek ve tahmin değerlerinden MAE hesaplar."""
    return mean_absolute_error(y_true, y_pred)  # Ortalama mutlak hata

def calculate_r2(y_true, y_pred):
    """Gerçek ve tahmin değerlerinden R² skorunu hesaplar."""
    return r2_score(y_true, y_pred)  # R-kare skoru

def get_residuals(y_true, y_pred):
    """Residual değerleri çıkarır (dashboard için input)."""
    return np.array(y_true) - np.array(y_pred)  # Gerçek - tahmin farkı
