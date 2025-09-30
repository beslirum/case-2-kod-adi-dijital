from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingModel:
    def __init__(
        self,
        config: Optional[Dict] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.config = config or {}
        self.model = GradientBoostingRegressor(**self.config)
        self.feature_names = list(feature_names) if feature_names else None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.feature_names is None:
            self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        return self.model.score(X, y)

    def get_model_params(self) -> Dict[str, Any]:
        return self.model.get_params(deep=True)

    def set_model_params(self, **params) -> "GradientBoostingModel":
        """Optionally update parameters after init (before fit)."""
        self.model.set_params(**params)
        self.config.update(params)
        return self

    def summary(self) -> Dict[str, Any]:
        return {
            "estimator": "GradientBoostingRegressor",
            "config": self.config,
            "n_features": (
                len(self.feature_names) if self.feature_names is not None else None
            ),
            "fitted": self._fitted,
        }