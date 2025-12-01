from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import backend_api

app = FastAPI(title="Helios Backend Wrapper")

class TrainConfig(BaseModel):
    epochs: int = 10
    data_source: str = "mock"
    learning_rate: float = 0.001

@app.get("/health")
def health():
    return backend_api.get_health()

@app.get("/api/health")
def api_health():
    return backend_api.get_health()

@app.get("/components")
def components():
    return backend_api.available_components()

@app.post("/train/{model_name}")
def train(model_name: str, cfg: TrainConfig):
    result = backend_api.start_training(model_name, cfg.dict())
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/predict/{model_name}")
def predict(model_name: str, payload: Dict[str, Any]):
    result = backend_api.predict(model_name, payload)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result
