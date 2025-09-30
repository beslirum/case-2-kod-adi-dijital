import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd

@dataclass
class MLflowLogger:
    enabled = True

    def __post_init__(self):
        self._active_run = None

    def start_run(
        self,
        *,
        run_name: str,
        model_name: str,
        description: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        self._active_run = mlflow.start_run(run_name=run_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param(
            "timestamp_utc",
            timestamp_utc or (datetime.utcnow().isoformat(timespec="seconds") + "Z"),
        )
        if description is not None:
            mlflow.log_param("description", description)

    def log_data_sample(
        self, X, y, *, sample_rows: int = 5, artifact_name: str = "data_sample.csv"
    ) -> None:
        if not self.enabled or self._active_run is None:
            return
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_ser = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
        sample = X_df.head(sample_rows).copy()
        sample.insert(len(sample.columns), "target", y_ser.head(sample_rows).values)

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, artifact_name)
            sample.to_csv(p, index=False)
            mlflow.log_artifact(p, artifact_path="samples")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        *,
        step: Optional[int] = None,
        split: Optional[str] = None,
    ) -> None:
        if not self.enabled or self._active_run is None or not metrics:
            return
        if split:
            metrics = {f"{split}/{k}": float(v) for k, v in metrics.items()}
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def end_run(self) -> None:
        if not self.enabled or self._active_run is None:
            return
        mlflow.end_run()
        self._active_run = None