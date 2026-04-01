from fastapi import FastAPI
import joblib
import json
import numpy as np
import os
import pandas as pd
import urllib.error
import urllib.request
import warnings
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def _tanim_api_origin() -> str:
    """Origin of the main FastAPI app (no /api/v1).

    TANIM_API_BASE_URL may be http://0.0.0.0:8000 (same host/port you use with uvicorn --host 0.0.0.0).
    Outbound HTTP clients cannot use 0.0.0.0 as the destination; we connect to 127.0.0.1 instead.
    """
    raw = os.environ.get("TANIM_API_BASE_URL", "http://0.0.0.0:8000").rstrip("/")
    if raw.endswith("/api/v1"):
        raw = raw[: -len("/api/v1")].rstrip("/")
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    base = urlunparse((scheme, netloc, "", "", "", "")).rstrip("/")
    return base or "http://127.0.0.1:8000"


def _soil_health_test_url() -> str:
    return f"{_tanim_api_origin()}/api/v1/test/"


def _crop_recommendation_url() -> str:
    return f"{_tanim_api_origin()}/api/v1/crop-recommendations/"


class PredictRequest(BaseModel):
    features: List[float]
    farm_id: Optional[str] = None


class FertilizerPredictRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    temperature: float
    ec: float
    moisture: float
    farm_id: Optional[str] = None


def _soil_health_payload(features: List[float], farm_id: str) -> Dict[str, Any]:
    return {
        "nitrogen": float(features[0]),
        "phosphorus": float(features[1]),
        "potassium": float(features[2]),
        "ph": float(features[3]),
        "salinity": 0.0,
        "temperature": float(features[4]) if len(features) > 4 else 0.0,
        "moisture": float(features[5]) if len(features) > 5 else 0.0,
        "farm_id": farm_id,
    }


def _sync_soil_health_test_to_api(features: List[float], farm_id: str) -> None:
    url = _soil_health_test_url()
    body = json.dumps(_soil_health_payload(features, farm_id)).encode("utf-8")

    def _request(method: str) -> urllib.request.Request:
        return urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )

    try:
        with urllib.request.urlopen(_request("POST"), timeout=15) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 409:
            try:
                with urllib.request.urlopen(_request("PUT"), timeout=15) as resp:
                    resp.read()
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
            ) as e2:
                print(f"Soil health test API sync (PUT) failed ({url}): {e2}")
        else:
            print(f"Soil health test API sync failed ({url}): {e}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"Soil health test API sync failed ({url}): {e}")


def _sync_crop_recommendation_to_api(
    farm_id: str, probabilities: List[Dict[str, Any]]
) -> None:
    url = _crop_recommendation_url()
    body = json.dumps({"farm_id": farm_id, "probabilities": probabilities}).encode(
        "utf-8"
    )
    post_req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(post_req, timeout=15) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 409:
            put_req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="PUT",
            )
            try:
                with urllib.request.urlopen(put_req, timeout=15) as resp:
                    resp.read()
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
            ) as e2:
                print(f"Crop recommendation API sync (PUT) failed ({url}): {e2}")
        else:
            print(f"Crop recommendation API sync failed ({url}): {e}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"Crop recommendation API sync failed ({url}): {e}")


app = FastAPI(title="ML Inference API")

# --- Crop model artifacts ---
MODEL_DIR = "model_artifacts"

model = joblib.load(os.path.join(MODEL_DIR, "lgbm_crop_model.pkl"))

label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
except FileNotFoundError:
    scaler = None

# --- Fertilizer model artifacts ---
FERT_MODEL_DIR = "model_fert_artifacts"

fert_model = joblib.load(os.path.join(FERT_MODEL_DIR, "lgbm_fert_model.pkl"))
fert_label_encoder = joblib.load(os.path.join(FERT_MODEL_DIR, "label_encoder.pkl"))

try:
    fert_scaler = joblib.load(os.path.join(FERT_MODEL_DIR, "scaler.pkl"))
except FileNotFoundError:
    fert_scaler = None

with open(os.path.join(FERT_MODEL_DIR, "metadata.json"), "r") as f:
    fert_metadata = json.load(f)


def _engineer_fert_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 11 engineered features expected by the fertilizer model."""
    eps = 1e-6
    df["NPK_sum"] = df["Nitrogen_Level"] + df["Phosphorus_Level"] + df["Potassium_Level"]
    df["N_ratio"] = df["Nitrogen_Level"] / (df["NPK_sum"] + eps)
    df["P_ratio"] = df["Phosphorus_Level"] / (df["NPK_sum"] + eps)
    df["K_ratio"] = df["Potassium_Level"] / (df["NPK_sum"] + eps)
    df["NP_ratio"] = df["Nitrogen_Level"] / (df["Phosphorus_Level"] + eps)
    df["NK_ratio"] = df["Nitrogen_Level"] / (df["Potassium_Level"] + eps)
    df["PK_ratio"] = df["Phosphorus_Level"] / (df["Potassium_Level"] + eps)
    df["pH_EC_interaction"] = df["Soil_pH"] * df["Electrical_Conductivity"]
    df["Moisture_Temp_interaction"] = df["Soil_Moisture"] * df["Temperature"]
    df["N_per_moisture"] = df["Nitrogen_Level"] / (df["Soil_Moisture"] + eps)
    df["EC_pH_diff"] = df["Electrical_Conductivity"] - df["Soil_pH"]
    return df


def classify_npk(nitrogen: float, phosphorus: float, potassium: float) -> Dict[str, str]:
    """Classify quantitative NPK values into qualitative levels (Low/Medium/High).

    Thresholds (single-crop trial):
        Nitrogen:   Low >= 60,  Medium >= 50,  High < 50
        Phosphorus: Low >= 60,  Medium >= 35,  High < 35
        Potassium:  Low >= 90,  Medium >= 60,  High < 60
    """

    def _level(value: float, low_thresh: float, med_thresh: float) -> str:
        if value >= low_thresh:
            return "Low"
        elif value >= med_thresh:
            return "Medium"
        else:
            return "High"

    return {
        "nitrogen": _level(nitrogen, 60, 50),
        "phosphorus": _level(phosphorus, 60, 35),
        "potassium": _level(potassium, 90, 60),
    }


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        print(f"Predict request received: {request.model_dump()}")
        features = request.features
        farm_id = request.farm_id

        if model is None:
            return {"status": "error", "message": "Model not available"}

        soil_sample = pd.DataFrame(
            [
                {
                    "OM_pct": features[0],
                    "P_ppm": features[1],
                    "K_ppm": features[2],
                    "Soil_pH": features[3],
                }
            ]
        )

        if scaler is not None:
            input_data = scaler.transform(soil_sample)
        else:
            input_data = soil_sample

        prediction = model.predict(input_data)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]

            if label_encoder is not None:
                crops = label_encoder.classes_
            elif hasattr(model, "classes_"):
                crops = model.classes_
            else:
                crops = [str(prediction)]
                probs = [1.0]

            probs = [float(p) for p in probs]

            crop_probs = list(zip(crops, probs))
            top_3 = sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]

            all_probs = sorted(crop_probs, key=lambda x: x[1], reverse=True)
            print("All crop probabilities:")
            for crop, prob in all_probs:
                print(f"  {crop}: {prob:.4f}")

            if label_encoder is not None:
                prediction_name = label_encoder.inverse_transform([prediction])[0]
            else:
                prediction_name = str(prediction)

            prediction_str = str(prediction_name)
            if farm_id:
                _sync_soil_health_test_to_api(features, farm_id)
                probabilities_payload = [
                    {"crop_class": str(crop), "probability": float(prob)}
                    for crop, prob in all_probs
                ]
                _sync_crop_recommendation_to_api(farm_id, probabilities_payload)

            npk_levels = classify_npk(
                nitrogen=features[0],
                phosphorus=features[1],
                potassium=features[2],
            )

            return {
                "status": "success",
                "message": "Prediction successful",
                "data": {
                    "prediction": prediction_str,
                    "npk_levels": npk_levels,
                    "probabilities": [
                        {"crop_class": crop, "probability": float(prob)}
                        for crop, prob in top_3
                    ],
                },
            }
        else:
            if label_encoder is not None:
                prediction_name = label_encoder.inverse_transform([prediction])[0]
            else:
                prediction_name = str(prediction)

            prediction_str = str(prediction_name)
            if farm_id:
                _sync_soil_health_test_to_api(features, farm_id)

            npk_levels = classify_npk(
                nitrogen=features[0],
                phosphorus=features[1],
                potassium=features[2],
            )

            return {
                "status": "success",
                "message": "Prediction successful",
                "data": {
                    "prediction": prediction_str,
                    "npk_levels": npk_levels,
                },
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
        }


@app.post("/predict/fertilizer")
def fertilizer(request: FertilizerPredictRequest):
    try:
        if fert_model is None:
            return {"status": "error", "message": "Fertilizer model not available"}

        soil_sample = pd.DataFrame(
            [
                {
                    "Nitrogen_Level": request.nitrogen,
                    "Phosphorus_Level": request.phosphorus,
                    "Potassium_Level": request.potassium,
                    "Soil_pH": request.ph,
                    "Temperature": request.temperature,
                    "Electrical_Conductivity": request.ec,
                    "Soil_Moisture": request.moisture,
                }
            ]
        )

        soil_sample = _engineer_fert_features(soil_sample)

        feature_order = fert_metadata["all_features"]
        input_data = soil_sample[feature_order]

        if fert_scaler is not None:
            input_data = fert_scaler.transform(input_data)

        prediction = fert_model.predict(input_data)[0]
        if hasattr(prediction, "item"):
            prediction = prediction.item()

        prediction_name = (
            fert_label_encoder.inverse_transform([prediction])[0]
            if fert_label_encoder is not None
            else str(prediction)
        )

        result: Dict[str, Any] = {
            "prediction": str(prediction_name),
        }

        if hasattr(fert_model, "predict_proba"):
            probs = fert_model.predict_proba(input_data)[0]
            classes = (
                fert_label_encoder.classes_
                if fert_label_encoder is not None
                else fert_model.classes_
            )
            crop_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
            result["probabilities"] = [
                {"fertilizer_class": str(c), "probability": float(p)}
                for c, p in crop_probs[:3]
            ]

        return {
            "status": "success",
            "message": "Fertilizer recommendation successful",
            "data": result,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Fertilizer prediction failed: {str(e)}",
        }
