"""FastAPI serving layer for the fashion recommender."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .config import Settings
from .logging_utils import configure_logging
from .pipeline import RecommenderPipeline, recommend_for_history

configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Built once and reused across requests so the trained two-tower model and
    # FAISS index are loaded from disk a single time, not on every call.
    _state["pipeline"] = RecommenderPipeline(Settings())
    yield
    _state.clear()


app = FastAPI(
    title="Multimodal Fashion Recommender",
    description="Two-tower sequence-aware retrieval over the Amazon Fashion catalog.",
    version="0.1.0",
    lifespan=lifespan,
)


class RecommendRequest(BaseModel):
    history: list[str] = Field(
        ...,
        min_length=1,
        description="Product titles or category keywords representing a user's history.",
    )
    top_k: int = Field(default=5, ge=1, le=50)


class Recommendation(BaseModel):
    rank: int
    item_index: int
    title: str
    categories: str
    score: float


class RecommendResponse(BaseModel):
    history: list[str]
    recommendations: list[Recommendation]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, Any]:
    pipeline: RecommenderPipeline = _state["pipeline"]
    return pipeline.validate_artifacts().__dict__


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    pipeline: RecommenderPipeline = _state["pipeline"]

    if not pipeline.settings.vectors_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run `reco train` before serving requests.",
        )

    try:
        results = recommend_for_history(request.history, top_k=request.top_k, pipeline=pipeline)
    except Exception as exc:
        logger.exception("recommend_for_history failed")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations") from exc

    return RecommendResponse(
        history=request.history,
        recommendations=[Recommendation(**r) for r in results],
    )
