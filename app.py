from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
from typing import List, Any
import warnings
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class PredictRequest(BaseModel):
    features: List[float]

app = FastAPI(title="ML Inference API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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
            "N": features[0],
            "P": features[1], 
            "K": features[2],
            "ph": features[3],
            "temperature": features[4],
            "humidity": features[5]
        }])
        
        prediction = model.predict(soil_sample)[0]
        
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(soil_sample)[0]
            
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
