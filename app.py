from fastapi import FastAPI
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import List, Any
import warnings
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class PredictRequest(BaseModel):
    features: List[float]

app = FastAPI(title="ML Inference API")

import os

MODEL_DIR = "model_artifacts"

model = joblib.load(os.path.join(MODEL_DIR, "lgbm_crop_model.pkl"))

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

try:
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None


@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        features = request.features
        
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
            
            return {
                "status": "success",
                "message": "Prediction successful",
                "data": {
                    "prediction": str(prediction_name),
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
                
            return {
                "status": "success",
                "message": "Prediction successful",
                "data": {"prediction": str(prediction_name)}
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }
