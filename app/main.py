import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.dr import router as dr_router
from app.core.config import settings
from app.db.session import init_db
from app.schemas.dr import HealthResponse
from app.services.dr_service import DiabeticRetinopathyService

logger = logging.getLogger("medical_ai_platform")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    service = DiabeticRetinopathyService(settings=settings)
    try:
        service.load()
        logger.info(
            "Loaded model '%s' on device '%s'.",
            settings.model_name,
            service.device,
        )
    except FileNotFoundError as exc:
        logger.warning("Startup warning (model not loaded): %s", exc)
    app.state.dr_service = service
    yield
    service.unload()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Research & clinical decision-support API for Diabetic Retinopathy grading "
        "with calibrated confidence, specialist triage flags, and Grad-CAM++ explainability."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(request: Request):
    service: DiabeticRetinopathyService | None = getattr(
        request.app.state, "dr_service", None
    )
    loaded = bool(service and service.is_ready)
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_name=settings.model_name,
        device=str(service.device) if service else "unknown",
        version=settings.app_version,
    )


app.include_router(dr_router, prefix=settings.api_v1_prefix)
