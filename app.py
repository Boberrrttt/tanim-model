from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="ML Inference API")

# Load model at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: list[float]):
    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]
    label = label_encoder.inverse_transform([pred])[0]

    return {
        "prediction": label
    }
