from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from mlflow_logger import MLflowLogger

class ModelTrainer:
    def __init__(
        self,
        model,
        params: Dict[str, Any] = None,
        metrics: Dict[str, Callable] = None,
        mlflow_logger: Optional[MLflowLogger] = None,
    ):
        self.model = model
        self.params = params or {}
        self.metrics = metrics or {}
        self.fitted_ = False
        self.logger = mlflow_logger

    def _apply_params(self):
        """Apply params to sklearn estimators or your wrappers."""
        if not self.params:
            return
        if hasattr(self.model, "set_params"):
            self.model.set_params(**self.params)
        elif hasattr(self.model, "set_model_params"):
            self.model.set_model_params(**self.params)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "set_params"):
            self.model.model.set_params(**self.params)

    def train(
        self,
        X,
        y,
        log=False,
        run_name=None,
        model_name=None,
        description=None,
        sample_rows=5,
    ) -> None:
        self._apply_params()
        if log:
            if self.logger is None:
                self.logger = MLflowLogger(enabled=True)
            if not run_name or not model_name:
                raise ValueError("Please provide both 'run_name' and 'model_name'.")
            self.logger.start_run(
                run_name=run_name, model_name=model_name, description=description
            )
            self.logger.log_data_sample(X, y, sample_rows=sample_rows)
        self.model.fit(X, y)
        self.fitted_ = True

    def predict(self, X):
        if not self.fitted_:
            raise RuntimeError("Model is not trained yet. Call train(X, y) first.")
        return self.model.predict(X)

    def evaluate(
        self, X, y, *, split: Optional[str] = None, step: Optional[int] = None
    ) -> Dict[str, float]:
        if not self.fitted_:
            raise RuntimeError("Model is not trained yet. Call train(X, y) first.")
        y_pred = self.predict(X)
        results = {
            name: float(metric(y, y_pred)) for name, metric in self.metrics.items()
        }
        if self.logger is not None:
            self.logger.log_metrics(results, step=step, split=split)
        return results

    def evaluate_many(
        self,
        datasets: Mapping[str, Tuple[Any, Any]],
        *,
        step: Optional[int] = None,
        end_run: bool = True
    ) -> Dict[str, Dict[str, float]]:
        if not self.fitted_:
            raise RuntimeError("Model is not trained yet. Call train(X, y) first.")
        out = {}
        for split, (X, y) in datasets.items():
            out[split] = self.evaluate(X, y, split=split, step=step)
        if end_run and self.logger is not None:
            self.logger.end_run()
        return out