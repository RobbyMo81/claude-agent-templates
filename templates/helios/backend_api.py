"""Programmatic backend API wrapper for Helios.

Provides lightweight functions to interact with backend components without running an HTTP server.

Functions:
- get_health(): return health/status dict
- start_training(model_name, config): start training via trainer if available
- get_model_info(model_name): return model metadata from memory_store if available
- predict(model_name, input_data): run prediction via agent if available
- available_components(): report which backend modules are importable

This wrapper is defensive: if components are not importable, it returns sensible mock responses.
"""
from datetime import datetime
import importlib
from typing import Any, Dict, Optional

# Try to import backend components if they exist in the workspace root
_components = {}
for name in [
    "trainer",
    "memory_store",
    "agent",
    "metacognition",
    "decision_engine",
    "cross_model_analytics",
]:
    try:
        _components[name] = importlib.import_module(name)
    except Exception:
        _components[name] = None


def available_components() -> Dict[str, bool]:
    """Return which backend components are importable."""
    return {k: v is not None for k, v in _components.items()}


def get_health() -> Dict[str, Any]:
    """Return a health/status dict. Use memory_store if available as persistent check."""
    now = datetime.utcnow().isoformat() + "Z"
    comp = available_components()

    # If memory_store module provides a health check, try it
    if _components.get("memory_store"):
        try:
            ms = _components["memory_store"].MemoryStore
            # instantiate a short-lived store if constructor exists
            try:
                store = ms("helios_memory.db")
                stats = store.get_memory_statistics() if hasattr(store, "get_memory_statistics") else {}
                return {
                    "status": "healthy",
                    "service": "helios-backend",
                    "timestamp": now,
                    "components": comp,
                    "memory_stats": stats,
                }
            except Exception:
                # fallback if MemoryStore constructor fails
                return {"status": "healthy", "service": "helios-backend", "timestamp": now, "components": comp}
        except Exception:
            pass

    # Default mock health
    return {"status": "healthy", "service": "helios-backend", "timestamp": now, "components": comp}


def start_training(model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Start a training job via trainer.ModelTrainer if available.

    Returns a job info dict or an error dict.
    """
    config = config or {}
    trainer_mod = _components.get("trainer")
    if not trainer_mod:
        return {"error": "trainer module not available"}

    try:
        Trainer = getattr(trainer_mod, "ModelTrainer", None)
        if Trainer is None:
            return {"error": "ModelTrainer class not found in trainer module"}

        # Instantiate a trainer. If constructor expects model_dir or memory_store, try best-effort.
        try:
            trainer = Trainer()
        except TypeError:
            # try with memory_store if available
            ms = _components.get("memory_store")
            trainer = Trainer(memory_store=ms.MemoryStore("helios_memory.db") if ms else None)

        job_info = trainer.start_training_job(model_name=model_name, data_source=config.get("data_source", "mock"), config_override=config)
        return job_info
    except Exception as e:
        return {"error": str(e)}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model metadata from memory_store if available."""
    ms_mod = _components.get("memory_store")
    if not ms_mod:
        return {"error": "memory_store not available"}

    try:
        store = ms_mod.MemoryStore("helios_memory.db")
        meta = store.get_model_metadata(model_name)
        if not meta:
            return {"error": f"model not found: {model_name}"}
        return meta
    except Exception as e:
        return {"error": str(e)}


def predict(model_name: str, input_data: Any = None) -> Dict[str, Any]:
    """Run prediction via agent.MLPowerballAgent if available.

    If agent or model not available, return an error dict.
    """
    agent_mod = _components.get("agent")
    if not agent_mod:
        return {"error": "agent module not available"}

    try:
        Agent = getattr(agent_mod, "MLPowerballAgent", None)
        if Agent is None:
            return {"error": "MLPowerballAgent not found in agent module"}

        agent = Agent(model_dir="models")
        if not agent.load_model(model_name):
            return {"error": f"failed to load model: {model_name}"}

        # If input_data is None, attempt to load mock historical data from trainer.DataPreprocessor
        if input_data is None and _components.get("trainer"):
            try:
                dp = getattr(_components["trainer"], "DataPreprocessor", None)
                if dp:
                    pre = dp()
                    input_data = pre.load_historical_data("mock")
            except Exception:
                input_data = {}

        preds = agent.predict(input_data)
        return preds if isinstance(preds, dict) else {"predictions": preds}
    except Exception as e:
        return {"error": str(e)}
