"""
FluxRoute – FastAPI server with OpenEnv endpoints.

Endpoints:
    POST /reset   — reset environment, returns Observation
    POST /step    — take action, returns StepResult
    GET  /state   — returns current State
    GET  /health  — health check
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment.env import RoutingEnv
from environment.models import Action, Observation, State, StepResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fluxroute.server")

app = FastAPI(
    title="FluxRoute",
    description="OpenEnv-compliant adaptive RL network routing environment",
    version="1.0.0",
)

# single global env instance (stateful per session)
env = RoutingEnv()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_static_mesh"
    seed: int = 42


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "environment": "FluxRoute", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        logger.info(f"Reset: task={req.task_id} seed={req.seed}")
        return obs
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Step failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=State)
def state():
    try:
        return env.state()
    except Exception as e:
        logger.error(f"State failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
