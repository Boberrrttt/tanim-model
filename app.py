from fastapi import FastAPI
import joblib
import json
import numpy as np
import os
import pandas as pd
import urllib.error
import urllib.request
import warnings
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def _soil_health_test_url() -> str:
    base = os.environ.get("TANIM_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    if base.endswith("/api/v1"):
        base = base[: -len("/api/v1")].rstrip("/")
    return f"{base}/api/v1/test/"


class PredictRequest(BaseModel):
    features: List[float]
    farm_id: Optional[str] = None


def _soil_health_payload(features: List[float], farm_id: str, classification: str) -> Dict[str, Any]:
    return {
        "nitrogen": float(features[0]),
        "phosphorus": float(features[1]),
        "potassium": float(features[2]),
        "ph": float(features[3]),
        "salinity": 0.0,
        "temperature": float(features[4]) if len(features) > 4 else 0.0,
        "moisture": float(features[5]) if len(features) > 5 else 0.0,
        "farm_id": farm_id,
        "classification": classification,
    }


def _sync_soil_health_test_to_api(features: List[float], farm_id: str, classification: str) -> None:
    url = _soil_health_test_url()
    body = json.dumps(_soil_health_payload(features, farm_id, classification)).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"Soil health test API sync failed ({url}): {e}")

app = FastAPI(title="ML Inference API")

MODEL_DIR = "model_artifacts"

model = joblib.load(os.path.join(MODEL_DIR, "lgbm_crop_model.pkl"))

label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
except FileNotFoundError:
    scaler = None


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
        
        soil_sample = pd.DataFrame([{
            "OM_pct": features[0],
            "P_ppm": features[1], 
            "K_ppm": features[2],
            "Soil_pH": features[3]
        }])
        
        if scaler is not None:
            # We assume it expects a DataFrame or 2D array and we pass it
            input_data = scaler.transform(soil_sample)
        else:
            input_data = soil_sample
            
        prediction = model.predict(input_data)[0]
        
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_data)[0]
            
            if label_encoder is not None:
                crops = label_encoder.classes_
            elif hasattr(model, 'classes_'):
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
                _sync_soil_health_test_to_api(features, farm_id, prediction_str)

            return {
                "status": "success",
                "message": "Prediction successful",
                "data": {
                    "prediction": prediction_str,
                    "probabilities": [
                        {"crop_class": crop, "probability": float(prob)} 
                        for crop, prob in top_3
                    ]
                }
            }
        else:
            if label_encoder is not None:
                prediction_name = label_encoder.inverse_transform([prediction])[0]
            else:
                prediction_name = str(prediction)

            prediction_str = str(prediction_name)
<<<<<<< HEAD
            
            _sync_soil_health_test_to_api(features, farm_id, prediction_str)
=======
            if farm_id:
                _sync_soil_health_test_to_api(features, farm_id, prediction_str)
>>>>>>> f25bf0476500eb849d02606305e73e7fe43ebc4c

            return {
                "status": "success",
                "message": "Prediction successful",
                "data": {"prediction": prediction_str}
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }
