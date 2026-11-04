import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from app.runtime import Runtime, BACKENDS
from app.config import settings
import json
from pathlib import Path

runtime = None
executor = ThreadPoolExecutor(max_workers=1)
gate = asyncio.Semaphore(int(os.environ.get("MAX_OUTSTANDING", "64")))

@asynccontextmanager
async def lifespan(app):
    global runtime
    runtime = Runtime()
    yield
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

class Request(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    backend: str | None = None

@app.get("/healthz")
async def health():
    return {"ready": runtime is not None, **(runtime.metadata if runtime else {}), **(runtime.stats() if runtime else {})}

@app.post("/v1/audio/speech")
async def speech(request: Request):
    request.backend = request.backend or settings()['serve_backend']
    if request.backend not in BACKENDS:
        raise HTTPException(422, "Unknown backend")
    if request.backend == 'bf16_aoti':
        if runtime.aoti is None:
            raise HTTPException(503, 'No compiled package. Run a benchmark or build it on the allocated GPU.')
        allowed = os.environ.get('BENCHMARK_MODE') == '1' or settings()['allow_unqualified_aoti']
        if not allowed:
            raise HTTPException(409, 'AOTI is experimental. Set allow_unqualified_aoti=true only after reviewing the numerical and ASR report.')
    if gate.locked():
        raise HTTPException(429, "Request queue is full")
    admitted = time.perf_counter()
    def run():
        queue_s = time.perf_counter() - admitted
        body, metrics = runtime.synthesize(request.text, request.backend)
        return body, {**metrics, "queue_s": queue_s}
    async with gate:
        try:
            body, metrics = await asyncio.get_running_loop().run_in_executor(executor, run)
        except AssertionError as exc:
            raise HTTPException(422, str(exc)) from exc
    return Response(body, media_type="audio/wav", headers={
        "X-Process-Seconds": str(metrics["process_s"]),
        "X-Queue-Seconds": str(metrics["queue_s"]),
        "X-Audio-Seconds": str(metrics["audio_s"]),
        "X-Backend": request.backend})
