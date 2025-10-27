"""
Production API service exposing inference endpoints.
Version: 3.1
"""

import os
import sys
from pathlib import Path
from tempfile import gettempdir


def _configure_prometheus_dir() -> Path:
    """Ensure Prometheus multiprocess metrics directory exists on all platforms."""
    env_value = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if env_value:
        target = Path(env_value)
    else:
        target = Path(gettempdir()) / "psuite-metrics"
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(target)

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fallback = Path(gettempdir()) / "psuite-metrics"
        if target != fallback:
            target = fallback
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(target)
            target.mkdir(parents=True, exist_ok=True)
        else:
            raise RuntimeError(f"Unable to initialise Prometheus multiprocess directory: {target}") from exc
    return target


PROMETHEUS_DIR = _configure_prometheus_dir()

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator, model_validator
import uvicorn
import time
import logging
import json
import signal
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio
import numpy as np
import uuid
from contextvars import ContextVar

from physics import FRAMEWORKS, create_model

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)

# ============================================================================
# LOGGING
# ============================================================================

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        return json.dumps(log_data)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("ProductionAPI")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

if PROMETHEUS_DIR.exists():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    logger.info("Multiprocess metrics enabled")
else:
    from prometheus_client import REGISTRY

    registry = REGISTRY

request_count = Counter(
    'http_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

prediction_latency = Histogram(
    'prediction_duration_seconds',
    'Prediction latency',
    ['model'],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
    registry=registry
)

active_requests = Gauge(
    'active_requests',
    'Active requests',
    registry=registry
)

error_count = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'endpoint'],
    registry=registry
)

gpu_memory_bytes = Gauge(
    'gpu_memory_bytes',
    'GPU memory usage',
    ['device'],
    registry=registry
)

api_info = Info('api', 'API info', registry=registry)
api_info.info({
    'version': '3.1',
    'model': 'MDNO v5.3',
})

# ============================================================================
# REQUEST TRACKING
# ============================================================================

class RequestTracker:
    """Thread-safe request tracking and GPU metrics."""

    def __init__(self):
        self._count = 0
        self._lock = asyncio.Lock()
        self._gpu_memory = {}
    
    async def increment(self):
        async with self._lock:
            self._count += 1
            active_requests.set(self._count)
    
    async def decrement(self):
        async with self._lock:
            self._count = max(0, self._count - 1)
            active_requests.set(self._count)
    
    def get_count(self) -> int:
        return self._count
    
    def update_gpu_memory(self):
        """Update GPU metrics safely."""
        if not torch.cuda.is_available():
            return
        try:
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i)
                gpu_memory_bytes.labels(device=f'cuda:{i}').set(mem)
                self._gpu_memory[f'cuda:{i}'] = mem
        except RuntimeError as exc:
            logger.warning("Failed to collect GPU memory metrics: %s", exc)

    def get_gpu_memory(self, device: str) -> float:
        return self._gpu_memory.get(device, 0.0) / 1e9

request_tracker = RequestTracker()

# ============================================================================
# CONTEXT
# ============================================================================

class RequestContext:
    def __init__(self):
        self.request_id: str = ""
        self.start_time: float = 0.0

request_context: ContextVar[RequestContext] = ContextVar('request_context')

def get_request_id() -> str:
    ctx = request_context.get(None)
    return ctx.request_id if ctx else "unknown"

# ============================================================================
# MODELS (Pydantic v2 syntax)
# ============================================================================

class TensorShape(BaseModel):
    dimensions: list[int] = Field(..., min_length=1, max_length=5)
    
    @field_validator('dimensions')
    @classmethod
    def validate_dims(cls, v):
        if any(d <= 0 or d > 1000 for d in v):
            raise ValueError("Dimensions must be in [1, 1000]")
        total = 1
        for d in v:
            total *= d
        if total > 1_000_000:
            raise ValueError("Total size > 1M elements")
        return v

class PredictionRequest(BaseModel):
    model_type: str = Field(..., pattern="^(rtno|mdno)$")
    inputs: Dict[str, List[float]]
    shape: TensorShape
    return_uncertainty: bool = False
    batch_size: int = Field(1, ge=1, le=64)
    timeout_seconds: float = Field(30.0, gt=0, le=300)

    @model_validator(mode='after')
    def validate_inputs(self):
        if not self.inputs:
            raise ValueError("Inputs must not be empty")

        total_size = int(np.prod(self.shape.dimensions))
        for key, values in self.inputs.items():
            if len(values) != total_size:
                raise ValueError(f"Input {key} size mismatch")

        model = self.model_type.lower()
        if model == 'rtno':
            required = {"temperature", "pressure"}
            missing = sorted(required.difference(self.inputs))
            if missing:
                formatted = ", ".join(missing)
                raise ValueError(
                    f"RTNO inputs require temperature and pressure fields (missing: {formatted})"
                )
        elif model == 'mdno':
            allowed = {"micro", "meso", "macro"}
            if not any(name in allowed for name in self.inputs):
                raise ValueError(
                    "MDNO inputs require at least one of micro/meso/macro"
                )
        return self

class PredictionResponse(BaseModel):
    request_id: str
    outputs: Dict[str, List[float]]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    uptime_seconds: float
    version: str

class MetricsResponse(BaseModel):
    active_requests: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    gpu_memory_gb: float
    uptime_seconds: float

# ============================================================================
# AUTH
# ============================================================================

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens: float = float(capacity)
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class APIKeyAuth:
    def __init__(self, valid_keys: Dict[str, Dict[str, Any]]):
        self.valid_keys = valid_keys
        self.rate_limiters = {
            key: TokenBucket(info.get("rate_limit", 100), info.get("rate_limit", 100) * 2)
            for key, info in valid_keys.items()
        }
    
    async def __call__(self, request: Request, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
        api_key = credentials.credentials
        
        if api_key not in self.valid_keys:
            error_count.labels(error_type='auth', endpoint=request.url.path).inc()
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        if not await self.rate_limiters[api_key].consume():
            error_count.labels(error_type='rate_limit', endpoint=request.url.path).inc()
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        return api_key

# ============================================================================
# INFERENCE
# ============================================================================

class InferenceEngine:
    def __init__(self, models: Dict[str, nn.Module], device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, nn.Module] = {}
        self.latencies: Dict[str, List[float]] = {}
        self._global_latencies: List[float] = []
        self.max_samples = 10000

        for name, model in models.items():
            key = name.lower()
            self.models[key] = model.to(self.device)
            self.models[key].eval()
            self.latencies[key] = []

        logger.info("Engine initialized with models: %s on %s", list(self.models.keys()), self.device)

    async def predict(self, model_key: str, inputs: Dict[str, torch.Tensor], timeout: float = 30.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        model = self._get_model(model_key)
        await request_tracker.increment()

        try:
            start = time.perf_counter()

            try:
                outputs = await asyncio.wait_for(
                    self._run_inference(model, inputs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                error_count.labels(error_type='timeout', endpoint='/predict').inc()
                raise HTTPException(status_code=408, detail="Timeout")

            duration = time.perf_counter() - start
            prediction_latency.labels(model=model_key).observe(duration)

            self.latencies[model_key].append(duration)
            self._global_latencies.append(duration)
            if len(self.latencies[model_key]) > self.max_samples:
                self.latencies[model_key].pop(0)
            if len(self._global_latencies) > self.max_samples:
                self._global_latencies.pop(0)

            request_tracker.update_gpu_memory()

            return outputs

        except HTTPException:
            raise
        except ValueError as exc:
            error_count.labels(error_type='validation', endpoint='/predict').inc()
            logger.warning("Validation failure during prediction: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            error_count.labels(error_type='internal', endpoint='/predict').inc()
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Internal error")

        finally:
            await request_tracker.decrement()

    async def _run_inference(self, model: nn.Module, inputs: Dict[str, torch.Tensor]):
        return await asyncio.to_thread(self._run_inference_sync, model, inputs)

    def _run_inference_sync(self, model: nn.Module, inputs: Dict[str, torch.Tensor]):
        inputs_device = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(inputs_device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return outputs
    
    def get_latency_stats(self):
        if not self._global_latencies:
            return {'p50': 0, 'p95': 0, 'p99': 0}
        sorted_lat = sorted(self._global_latencies)
        n = len(sorted_lat)
        return {
            'p50': sorted_lat[int(n * 0.50)],
            'p95': sorted_lat[int(n * 0.95)],
            'p99': sorted_lat[min(int(n * 0.99), n - 1)]
        }

    def _get_model(self, model_key: str) -> nn.Module:
        key = model_key.lower()
        if key not in self.models:
            raise HTTPException(status_code=404, detail=f"Model '{model_key}' not available")
        return self.models[key]

# ============================================================================
# API
# ============================================================================

class ProductionAPI:
    def __init__(
        self,
        models: Dict[str, nn.Module],
        api_keys: Optional[Dict[str, Dict[str, Any]]] = None,
        allowed_origins: Optional[List[str]] = None,
    ):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info("API starting")
            self.start_time = time.time()
            yield
            logger.info("API shutting down")
            await self.shutdown()
        
        self.app = FastAPI(
            title="Atmospheric Model API",
            version="3.1.0",
            lifespan=lifespan
        )
        
        # CORS
        cors_origins = allowed_origins or os.getenv("CORS_ORIGINS", "").split(",")
        if not cors_origins or cors_origins == [""]:
            cors_origins = ["http://localhost:3000", "http://localhost:8000"]
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        self.auth = APIKeyAuth(api_keys or {}) if api_keys else None
        self.engine = InferenceEngine(models)
        self.available_models = sorted(models.keys())
        self.start_time = time.time()
        self.shutdown_event = asyncio.Event()
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_signals()
        
        logger.info("API initialized with models: %s", self.available_models)
    
    def _setup_signals(self):
        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} received")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def shutdown(self):
        logger.info("Graceful shutdown starting")
        self.shutdown_event.set()
        
        wait_time = 0
        while request_tracker.get_count() > 0 and wait_time < 30:
            logger.info(f"Waiting for {request_tracker.get_count()} requests...")
            await asyncio.sleep(1)
            wait_time += 1
        
        # Mark process dead for multiprocess metrics
        if PROMETHEUS_DIR.exists():
            multiprocess.mark_process_dead(os.getpid())
        
        logger.info("Shutdown complete")
    
    def _normalize_endpoint(self, path: str) -> str:
        if path.startswith("/predict"):
            return "/predict"
        elif path.startswith("/health"):
            return "/health"
        elif path.startswith("/metrics"):
            return "/metrics"
        return "/other"
    
    def _setup_middleware(self):
        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            ctx = RequestContext()
            ctx.request_id = str(uuid.uuid4())
            ctx.start_time = time.perf_counter()
            token = request_context.set(ctx)

            endpoint = self._normalize_endpoint(request.url.path)
            status_code = 500

            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            except Exception:
                logger.exception("Request failed")
                raise
            finally:
                request_count.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status=str(status_code)
                ).inc()
                request_context.reset(token)
    
    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"service": "Atmospheric Model API", "version": "3.1.0"}
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow().isoformat(),
                model_loaded=True,
                uptime_seconds=time.time() - self.start_time,
                version="3.1.0"
            )
        
        @self.app.get("/health/ready")
        async def readiness():
            if self.shutdown_event.is_set():
                raise HTTPException(status_code=503, detail="Shutting down")
            return {"status": "ready"}
        
        @self.app.get("/health/live")
        async def liveness():
            return {"status": "alive"}
        
        @self.app.get("/metrics")
        async def metrics():
            return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
        
        @self.app.get("/metrics/summary", response_model=MetricsResponse)
        async def metrics_summary():
            request_tracker.update_gpu_memory()
            stats = self.engine.get_latency_stats()
            return MetricsResponse(
                active_requests=request_tracker.get_count(),
                latency_p50_ms=stats['p50'] * 1000,
                latency_p95_ms=stats['p95'] * 1000,
                latency_p99_ms=stats['p99'] * 1000,
                gpu_memory_gb=request_tracker.get_gpu_memory('cuda:0'),
                uptime_seconds=time.time() - self.start_time
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionRequest,
            api_key: Optional[str] = Depends(self.auth) if self.auth else None
        ):
            request_id = get_request_id()
            
            try:
                inputs = {}
                for key, value in request.inputs.items():
                    try:
                        tensor = torch.tensor(value, dtype=torch.float32)
                        tensor = tensor.reshape(request.shape.dimensions)
                    except (TypeError, RuntimeError) as exc:
                        raise ValueError(f"Invalid tensor for '{key}': {exc}") from exc
                    inputs[key] = tensor
                
                model_key = request.model_type.lower()
                outputs = await self.engine.predict(model_key, inputs, request.timeout_seconds)
                
                outputs_list = {
                    k: v.cpu().numpy().flatten().tolist()
                    for k, v in outputs.items() if torch.is_tensor(v)
                }
                
                return PredictionResponse(
                    request_id=request_id,
                    outputs=outputs_list,
                    metadata={
                        'model': model_key,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
            
            except ValueError as e:
                error_count.labels(error_type='validation', endpoint='/predict').inc()
                raise HTTPException(status_code=400, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        uvicorn.run(self.app, host=host, port=port, workers=workers, log_level="info")

# ============================================================================
# MAIN
# ============================================================================

def load_default_models() -> Dict[str, nn.Module]:
    """Instantiate default physics models for the service."""
    models: Dict[str, nn.Module] = {}
    models["mdno"] = create_model("mdno", config_kwargs={"use_radiative_transfer": True})
    models["rtno"] = create_model("rtno")
    return models

def main():
    print("="*80)
    print("Production API v3.1")
    print("="*80)
    
    models = load_default_models()
    
    api = ProductionAPI(
        models=models,
        api_keys={
            "test-key-123": {"user_id": "test", "rate_limit": 100}
        },
        allowed_origins=["http://localhost:3000", "https://yourdomain.com"]
    )
    
    print("\n[OK] Production-ready features:")
    print("  - Multiprocess-safe metrics")
    print("  - No private Prometheus members")
    print("  - RequestTracker encapsulates GPU telemetry")
    print("  - Pydantic v2 syntax")
    print("  - Environment-driven CORS")
    print("  - Structured logging")
    print("  - Graceful shutdown with mark_process_dead()")
    print(f"  - Available models: {sorted(models.keys())}")
    print(f"  - Supported frameworks: {sorted(FRAMEWORKS)}")
    print("\nStarting server on http://0.0.0.0:8000")
    print("="*80)
    
    api.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

