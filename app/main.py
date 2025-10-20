from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from typing import Optional, Dict, Any, List

from ml.train import ensure_model
from ml.predict import predict_from_features
from ml.dataset import build_dataset

app = FastAPI(title="Investment Rules ML API")

# Static and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(BASE_DIR)
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
# Fix: serve static from app/static where our CSS and images live
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Cached metadata for dropdowns
SECTOR_OPTIONS: List[str] = []
YEAR_OPTIONS: List[int] = []


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def startup() -> None:
    # Prepare model
    ensure_model()
    # Load metadata from dataset
    try:
        df = build_dataset()
        sectors = sorted([s for s in df["Sector"].dropna().unique().tolist() if isinstance(s, str)])
        years = sorted(df["year"].dropna().unique().astype(int).tolist())
    except Exception:
        sectors, years = [], []
    global SECTOR_OPTIONS, YEAR_OPTIONS
    SECTOR_OPTIONS = sectors
    YEAR_OPTIONS = years


@app.get("/", response_class=HTMLResponse)
def index(request: Request, message: Optional[str] = None, result: Optional[Dict[str, Any]] = None) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sectors": SECTOR_OPTIONS,
            "years": YEAR_OPTIONS,
            "message": message,
            "result": result,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def form_predict(
    request: Request,
    year: Optional[int] = Form(default=None),
    rank: Optional[float] = Form(default=None),
    Sector: Optional[str] = Form(default=None),
    is_top_50: Optional[int] = Form(default=None),
    is_top_100: Optional[int] = Form(default=None),
    is_top_500: Optional[int] = Form(default=None),
) -> HTMLResponse:
    try:
        payload: Dict[str, Any] = {}
        if year is not None:
            payload["year"] = int(year)
        if rank is not None:
            payload["rank"] = float(rank)
        if Sector:
            payload["Sector"] = Sector
        if is_top_50 is not None and str(is_top_50) != "":
            payload["is_top_50"] = int(is_top_50)
        if is_top_100 is not None and str(is_top_100) != "":
            payload["is_top_100"] = int(is_top_100)
        if is_top_500 is not None and str(is_top_500) != "":
            payload["is_top_500"] = int(is_top_500)

        pred = predict_from_features(payload)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "sectors": SECTOR_OPTIONS,
                "years": YEAR_OPTIONS,
                "result": pred,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "sectors": SECTOR_OPTIONS,
                "years": YEAR_OPTIONS,
                "message": f"Error: {str(exc)}",
            },
            status_code=400,
        )


@app.post("/api/train")
def api_train() -> Dict[str, str]:
    ensure_model(force_retrain=True)
    return {"status": "trained"}


@app.post("/api/predict")
def api_predict(payload: Dict[str, Any]) -> JSONResponse:
    try:
        result = predict_from_features(payload)
        return JSONResponse({"ok": True, "prediction": result})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
