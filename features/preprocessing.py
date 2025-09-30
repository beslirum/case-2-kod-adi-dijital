import pandas as pd
import numpy as nd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#df = pd.read_csv(r"C:\Users\ybayraktar\Desktop\Yeni klasör\sample_submission.csv")
test_df = pd.read_csv(r"C:\\Users\\rbesli\\Desktop\\src\\data\\test.csv")
train_df = pd.read_csv(r"C:\\Users\\rbesli\\Desktop\\src\\data\\train.csv")



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
