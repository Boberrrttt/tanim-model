import joblib
import os
import warnings

import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class CropService:
    """Handles crop recommendation via the LightGBM model."""

    def __init__(self, model_dir: str = "model_artifacts"):
        self._model = joblib.load(os.path.join(model_dir, "lgbm_crop_model.pkl"))
        self._label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

        try:
            self._scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        except FileNotFoundError:
            self._scaler = None

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Run crop prediction and return top-3 probabilities.

        Parameters
        ----------
        features : [OM_pct, P_ppm, K_ppm, Soil_pH]

        Returns
        -------
        dict with keys: prediction, probabilities (all), top_3
        """
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

        input_data = (
            self._scaler.transform(soil_sample)
            if self._scaler is not None
            else soil_sample
        )

        raw_prediction = self._model.predict(input_data)[0]
        if hasattr(raw_prediction, "item"):
            raw_prediction = raw_prediction.item()

        prediction_name = str(
            self._label_encoder.inverse_transform([raw_prediction])[0]
        )

        # probabilities
        all_probs: List[Dict[str, Any]] = []
        top_3: List[Dict[str, Any]] = []

        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(input_data)[0]
            crops = self._label_encoder.classes_
            probs = [float(p) for p in probs]

            crop_probs = sorted(
                zip(crops, probs), key=lambda x: x[1], reverse=True
            )
            all_probs = [
                {"crop_class": str(c), "probability": p} for c, p in crop_probs
            ]
            top_3 = all_probs[:3]

        return {
            "prediction": prediction_name,
            "probabilities": all_probs,
            "top_3": top_3,
        }
