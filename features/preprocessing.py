import os
import pandas as pd
import numpy as nd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from typing import List, Optional

#df = pd.read_csv(r"C:\Users\ybayraktar\Desktop\Yeni klasör\sample_submission.csv")
data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))



X = train_df.drop("y", axis=1)
y = train_df["y"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("y_train shape:", y_train.shape)
print("y_valid shape:", y_valid.shape)

numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns

def standardScaler(df : pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_valid_scaled[numeric_cols] = scaler.transform(X_valid[numeric_cols])

class Preprocessor:
    def __init__(self, n_components: Optional[int] = None):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.encoding_dict = dict()

    def scale_features(
        self, X: pd.DataFrame, features_to_scale: List[str]
    ) -> pd.DataFrame:
        result = X.copy()
        result[features_to_scale] = self.scaler.fit_transform(X[features_to_scale])
        return result

    def pca_fit(
        self, X: pd.DataFrame, features: List, n_components=None
    ) -> pd.DataFrame:
        result = X[features].copy()

        for col in features:
            if result[col].dtypes == "object":
                print(
                    f"Detected a categorical column ({col}), applying LabelEncoding..."
                )
                self.encoding_dict[col] = LabelEncoder()
                result[col] = self.encoding_dict[col].fit_transform(result[col])

        result = pd.DataFrame(
            self.scale_features(result, result.columns),
            columns=result.columns,
            index=result.index,
        )
        self.pca.fit(X=result)

    def pca_transform(
        self, X: pd.DataFrame, first_n_components: Optional[int] = None
    ) -> pd.DataFrame:
        if len(self.pca.feature_names_in_) == 0:
            raise RuntimeError("Call pca_fit before pca_transform.")

        X_num = X[self.pca.feature_names_in_.tolist()].copy()
        for col in X_num.columns:
            if X_num[col].dtype == "object":
                if col in self.encoding_dict:
                    X_num[col] = self.encoding_dict[col].transform(X_num[col])
                else:
                    raise ValueError("Detected categorical columns...")

        Xs = self.scaler.transform(X_num)
        Z = self.pca.transform(Xs)

        k = (
            Z.shape[1]
            if first_n_components is None
            else min(first_n_components, Z.shape[1])
        )
        Z = Z[:, :k]
        cols = [f"PC{i+1}" for i in range(k)]
        return pd.DataFrame(Z, index=X.index, columns=cols)