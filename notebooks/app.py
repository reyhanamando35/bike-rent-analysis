from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi import Form

app = FastAPI()

# Load artifacts
artifacts = joblib.load("models_bundle.pkl")

# Models
models = {
    "lgbm": artifacts["lgbm"],
    "simple_linear": artifacts["simple_linear"],
    "multiple_linear": artifacts["multiple_linear"],
    "knn": artifacts["knn"]
}

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
    
# LightGBM page
templates = Jinja2Templates(directory="templates")


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



# -----------------------
# Model detail page
# -----------------------
@app.get("/model/{model_name}")
def model_page(request: Request, model_name: str):

    if model_name not in metrics:
        return {"error": "Model not found"}

    return templates.TemplateResponse(
        "model.html",
        {
            "request": request,
            "model_name": model_name,
            "metrics": metrics[model_name]
        }
    )


# -----------------------
# Comparison page
# -----------------------
@app.get("/comparison")
def comparison_page(request: Request):

    df = pd.DataFrame(metrics).T.reset_index()
    df.columns = ["Model", "RMSE", "R2"]

    return templates.TemplateResponse(
        "comparison.html",
        {
            "request": request,
            "table": df.to_dict(orient="records")
        }
    )
