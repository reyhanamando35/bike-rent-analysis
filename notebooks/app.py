from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi import Form
import json
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

# Load artifacts
artifacts = joblib.load("models_bundle.pkl")

# Models
models = {
    "lgbm": artifacts["lgbm"],
}
forecasting_model = joblib.load("lgbm_forecast.pkl")

# Preprocessing
numeric_scaler = artifacts["numeric_scaler"]
encoder = artifacts["encoder"]
knn_scaler = artifacts["knn_scaler"]
features = artifacts["features"]
metrics = artifacts["metrics"]

categorical_features = [
    "season", "yr", "mnth", "hr",
    "weekday", "workingday", "weathersit", "holiday"
]

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Homepage
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/model/lightgbm")
def lightgbm_page(request: Request):
    return templates.TemplateResponse(
        "lightgbm.html",
        {"request": request}
    )

@app.post("/predict/lightgbm")
def predict_lightgbm(
    temp: float = Form(...),
    atemp: float = Form(...),
    hum: float = Form(...),
    windspeed: float = Form(...),
    season: int = Form(...),
    yr: int = Form(...),
    mnth: int = Form(...),
    hr: int = Form(...),
    weekday: int = Form(...),
    workingday: int = Form(...),
    weathersit: int = Form(...),
    holiday: int = Form(...)
):
    # Build DataFrame
    df = pd.DataFrame([{
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "holiday": holiday
    }])

    # -------- Preprocessing --------
    X = numeric_scaler.transform(df)
    X[categorical_features] = X[categorical_features].astype("category")
    X = encoder.transform(X)

    # Enforce training column order
    X = X[features["lgbm_columns"]]

    # Predict
    prediction = models["lgbm"].predict(X)[0]

    return {
        "prediction": round(float(prediction), 2)
    }


# Forecasting Logic
forecasting_model = joblib.load("lgbm_forecast.pkl")
MODEL_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "day"
]

# ---------------- GET PAGE ----------------
@app.get("/forecasting")
def forecasting_page(request: Request):

    with open("static/forecast_metrics.json", "r") as f:
        metrics = json.load(f)

    return templates.TemplateResponse(
        "forecasting.html",
        {
            "request": request,
            "metrics": metrics,
            "feature_importance_img": "/static/feature_importance.png",
            "evaluation_img": "/static/forecast_evaluation.png",
        }
    )


def predict_manual(
    tanggal,
    jam,
    suhu,
    kelembaban,
    kondisi_cuaca,
    libur
):
    dt = pd.to_datetime(tanggal)

    # ===== Derived / Random features =====
    year = 0 if dt.year == 2011 else 1
    month = dt.month
    weekday = dt.weekday()
    day = dt.day

    # Random season (1–4) if not inferred
    if month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    elif month in [9, 10, 11]:
        season = 3
    else:
        season = 4

    # Random working day logic
    workingday = 1 if weekday < 5 and libur == 0 else 0

    # Random / default fillers
    atemp = suhu + np.random.uniform(-0.05, 0.05)
    windspeed = np.random.uniform(0.05, 0.5)

    # ===== Assemble full feature row =====
    row = {
        "season": season,
        "yr": year,
        "mnth": month,
        "hr": jam,
        "holiday": libur,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": kondisi_cuaca,
        "temp": suhu,
        "atemp": atemp,
        "hum": kelembaban,
        "windspeed": windspeed,
        "day": day
    }

    X = pd.DataFrame([[row[col] for col in MODEL_FEATURES]], columns=MODEL_FEATURES)

    prediction = forecasting_model.predict(X)[0]
    return max(0, int(prediction))


def forecast_full_day(
    target_date,
    suhu,
    kelembaban,
    kondisi_cuaca,
    libur,
    output_path
):
    hours = list(range(24))
    forecast_results = []

    for h in hours:
        p = predict_manual(
            target_date,
            h,
            suhu,
            kelembaban,
            kondisi_cuaca,
            libur
        )
        forecast_results.append(p)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(hours, forecast_results, alpha=0.6, label="Predicted Count (Bar)")
    plt.plot(hours, forecast_results, marker="o", linewidth=2, label="Predicted Count (Line)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Predicted Number of Rentals")
    plt.title(f"Bike Rental Forecast – {target_date}")
    plt.xticks(hours)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    return hours, forecast_results


@app.post("/model/forecasting")
def run_forecast(
    request: Request,
    date: str = Form(...),
    temp: float = Form(...),
    humidity: float = Form(...),
    weather: int = Form(...),
    holiday: int = Form(...)
):
    img_path = "static/user_forecast.png"

    hours, forecast_results = forecast_full_day(
        target_date=date,
        suhu=temp,
        kelembaban=humidity,
        kondisi_cuaca=weather,
        libur=holiday,
        output_path=img_path
    )

    with open("static/forecast_metrics.json", "r") as f:
        metrics = json.load(f)

    return templates.TemplateResponse(
        "forecasting.html",
        {
            "request": request,
            "metrics": metrics,
            "feature_importance_img": "/static/feature_importance.png",
            "evaluation_img": "/static/forecast_evaluation.png",
            "user_forecast_img": "/static/user_forecast.png",
            "forecast_values": zip(hours, forecast_results)
        }
    )
