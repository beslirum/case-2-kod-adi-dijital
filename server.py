import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from evaluation.metrics import calculate_mae, calculate_r2, get_residuals
from features import preprocessing

logger = logging.getLogger("ml_server")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Training Results API",
    version="0.1.0",
    description="Runs model training, evaluates train/test splits, and pushes results to Azure Blob Storage.",
)

SUPPORTED_MODELS: Dict[str, Any] = {
    "gradient_boosting": GradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
}


class TrainRequest(BaseModel):
    experiment_name: str = Field(default="default-experiment", min_length=1)
    model_type: str = Field(default="gradient_boosting")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    push_to_azure: bool = True


class TrainResponse(BaseModel):
    experiment_name: str
    model_type: str
    timestamp_utc: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    azure_blob_paths: Optional[Dict[str, str]] = None


class AzureResultUploader:
    def __init__(self, connection_string: str, container_name: str, base_path: str = "experiments") -> None:
        if not connection_string:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set.")
        cleaned_base = base_path.strip("/") if base_path else ""
        self.connection_string = connection_string
        self.container_name = container_name or "ml-results"
        self.base_path = cleaned_base or "experiments"
        self._service_client = None
        self._container_client = None

    def _import_blob_classes(self):
        try:
            from azure.storage.blob import BlobServiceClient, ContentSettings
        except ImportError as exc:
            raise RuntimeError("azure-storage-blob package is required to push results to Azure.") from exc
        return BlobServiceClient, ContentSettings

    def _get_container_client(self):
        if self._container_client is not None:
            return self._container_client
        BlobServiceClient, _ = self._import_blob_classes()
        service_client = self._service_client or BlobServiceClient.from_connection_string(self.connection_string)
        self._service_client = service_client
        container_client = service_client.get_container_client(self.container_name)
        if not container_client.exists():
            container_client.create_container()
        self._container_client = container_client
        return container_client

    def upload_metrics(self, experiment_name: str, timestamp_utc: str, payload: Dict[str, Any]) -> Dict[str, str]:
        _, ContentSettings = self._import_blob_classes()
        container_client = self._get_container_client()
        blob_root = f"{self.base_path}/{experiment_name}/{timestamp_utc}"
        metrics_blob = f"{blob_root}/metrics.json"
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        container_client.upload_blob(
            name=metrics_blob,
            data=data,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        return {"metrics": metrics_blob}


def _select_model(model_type: str, hyperparameters: Dict[str, Any]):
    if model_type not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model_type '{model_type}'. Supported types: {supported}")
    model_cls = SUPPORTED_MODELS[model_type]
    try:
        return model_cls(**hyperparameters)
    except TypeError as exc:
        raise ValueError(f"Invalid hyperparameters for model '{model_type}': {exc}") from exc


def _safe_float(value: float) -> float:
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def _summarize_residuals(residuals: np.ndarray) -> Dict[str, float]:
    return {
        "mean": _safe_float(np.mean(residuals)),
        "std": _safe_float(np.std(residuals)),
        "min": _safe_float(np.min(residuals)),
        "max": _safe_float(np.max(residuals)),
    }


def _build_metrics(y_true, y_pred) -> Dict[str, float]:
    residuals = get_residuals(y_true, y_pred)
    metrics = {
        "mae": _safe_float(calculate_mae(y_true, y_pred)),
        "r2": _safe_float(calculate_r2(y_true, y_pred)),
    }
    metrics.update({f"residual_{k}": v for k, v in _summarize_residuals(residuals).items()})
    return metrics


def _execute_training(model_type: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    model = _select_model(model_type, hyperparameters)
    X_train = preprocessing.X_train
    X_valid = preprocessing.X_valid
    y_train = preprocessing.y_train
    y_valid = preprocessing.y_valid

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_valid)

    return {
        "train_metrics": _build_metrics(y_train, train_pred),
        "test_metrics": _build_metrics(y_valid, test_pred),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def run_training(payload: TrainRequest) -> TrainResponse:
    hyperparameters = payload.hyperparameters or {}
    try:
        metrics_bundle = _execute_training(payload.model_type, hyperparameters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Training routine failed.")
        raise HTTPException(status_code=500, detail="Training routine failed.") from exc

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    response_body = {
        "experiment_name": payload.experiment_name,
        "model_type": payload.model_type,
        "timestamp_utc": timestamp_utc,
        "train_metrics": metrics_bundle["train_metrics"],
        "test_metrics": metrics_bundle["test_metrics"],
        "hyperparameters": hyperparameters,
    }

    azure_blob_paths: Optional[Dict[str, str]] = None
    if payload.push_to_azure:
        try:
            uploader = AzureResultUploader(
                os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
                os.getenv("AZURE_STORAGE_CONTAINER", "ml-results"),
                os.getenv("AZURE_STORAGE_BASE_PATH", "experiments"),
            )
            azure_blob_paths = uploader.upload_metrics(
                experiment_name=payload.experiment_name,
                timestamp_utc=timestamp_utc,
                payload=response_body,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Azure upload failed.")
            raise HTTPException(status_code=502, detail="Failed to push results to Azure.") from exc

    return TrainResponse(**response_body, azure_blob_paths=azure_blob_paths)
