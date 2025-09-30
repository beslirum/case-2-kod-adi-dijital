import matplotlib.pyplot as plt  
import seaborn as sns  
import shap  # Özellik önemliliği için SHAP
import pandas as pd  
from sklearn.decomposition import PCA  # PCA indirgeme için
#from src.visualization import plots

def plot_feature_distribution(df: pd.DataFrame, columns=None, kind="hist"):
    """Özellik dağılımlarını çizer (histogram veya KDE)."""
    if columns is None:  # Kolon belirtilmezse sayısal kolonları al
        columns = df.select_dtypes(include=["number"]).columns
    for col in columns:  # Her kolon için ayrı grafik çiz
        plt.figure(figsize=(6, 4))  # Grafik boyutu
        if kind == "hist":  # Histogram seçildiyse
            sns.histplot(df[col], kde=True, bins=30)  # Histogram + KDE
        elif kind == "kde":  # KDE seçildiyse
            sns.kdeplot(df[col], shade=True)  # KDE grafiği
        plt.title(f"Feature distribution: {col}")  # Başlık
        plt.show()  # Grafiği göster

def plot_correlation_heatmap(df: pd.DataFrame):
    """Korelasyon matrisi ısı haritası çizer."""
    plt.figure(figsize=(10, 8))  # Grafik boyutu
    corr = df.corr()  # Korelasyon matrisi hesapla
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)  # Isı haritası
    plt.title("Correlation Heatmap")  # Başlık
    plt.show()  # Grafiği göster


def plot_pca_scatter(X: pd.DataFrame, y=None, n_components=2):
    """PCA ile 2 boyuta indirgeme ve scatter plot çizer."""
    pca = PCA(n_components=n_components)  # PCA nesnesi oluştur
    X_pca = pca.fit_transform(X)  # PCA dönüşümü uygula
    plt.figure(figsize=(6, 5))  # Grafik boyutu
    if y is not None:  # y verilmişse renklendir
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
    else:  # y yoksa tek renk scatter
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    plt.xlabel("PCA 1")  # X ekseni etiketi
    plt.ylabel("PCA 2")  # Y ekseni etiketi
    plt.title("PCA Scatter Plot")  # Başlık
    plt.show()  # Grafiği göster

def plot_residuals(y_true, y_pred):
    """Tahmin hatalarının dağılım histogramını çizer."""
    residuals = y_true - y_pred  # Hataları hesapla
    plt.figure(figsize=(6, 4))  # Grafik boyutu
    sns.histplot(residuals, kde=True, bins=30)  # Histogram + KDE
    plt.title("Residuals Distribution")  # Başlık
    plt.xlabel("Residual")  # X ekseni etiketi
    plt.show()  # Grafiği göster

def plot_shap_values(model, X: pd.DataFrame, max_display=10):
    """SHAP değerleri ile özellik önemliliği görselleştirir."""
    explainer = shap.TreeExplainer(model)  # SHAP açıklayıcı oluştur
    shap_values = explainer.shap_values(X)  # SHAP değerlerini hesapla
    shap.summary_plot(shap_values, X, max_display=max_display)  # Özet grafiği çiz

###PCA EKLENECEK
# from src.visualization import plots dosyasında X_train çekilecek ve şu ekleneck:

'''
pca = PCA(n_components=2)
X_train_numeric = X_train_scaled[numeric_cols]
X_train_pca = pca.fit_transform(X_train_numeric.values)

plt.figure(figsize=(8,6))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - İlk 2 Bileşen")
plt.show()
print("PCA Varyans Oranı:", pca.explained_variance_ratio_)

'''


