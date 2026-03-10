import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agents.researcher_agent import lookup_researcher
from backend.agents.discovery_agent import discover_by_topic
from backend.agents.phd_pipeline_agent import find_emerging_researchers

app = FastAPI()

# CORS middleware — allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ResearcherQuery(BaseModel):
    query: str


class TopicQuery(BaseModel):
    topic: str


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------
async def batch_calls(
    items: list,
    func,
    batch_size: int = 2,
    delay: float = 2.0,
) -> list:
    results: list = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[func(item) for item in batch], return_exceptions=True
        )
        results.extend(batch_results)
        if i + batch_size < len(items):
            await asyncio.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# Preseeded cache (module-level variable, but NOT initialized at import)
# ---------------------------------------------------------------------------
_preseeded_cache: Optional[List[Dict[str, Any]]] = None

PRESEEDED_RESEARCHERS = [
    "Fei-Fei Li",
    "Ben Mildenhall",
    "Jonathan Barron",
    "Andreas Geiger",
    "Matthias Niessner",
    "Vincent Sitzmann",
]


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/api/researcher")
async def api_researcher(body: ResearcherQuery):
    try:
        result = await lookup_researcher(body.query)
        return result
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/discover")
async def api_discover(body: TopicQuery):
    try:
        results = await discover_by_topic(body.topic)
        return results
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/phd-pipeline")
async def api_phd_pipeline(body: TopicQuery):
    try:
        results = await find_emerging_researchers(body.topic)
        return results
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/preseeded")
async def api_preseeded():
    global _preseeded_cache
    try:
        if _preseeded_cache is not None:
            return _preseeded_cache

        results = await batch_calls(
            PRESEEDED_RESEARCHERS,
            lookup_researcher,
            batch_size=2,
            delay=2.0,
        )

        # Filter out exceptions
        valid_results: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict) and "error" not in r:
                valid_results.append(r)

        _preseeded_cache = valid_results
        return valid_results
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")


# Mount static files AFTER API routes so API routes take priority
app.mount("/", StaticFiles(directory="frontend"), name="frontend")
