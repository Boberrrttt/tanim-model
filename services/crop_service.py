import joblib
import os
import warnings

import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class CropService:
    """Handles crop recommendation via the LightGBM model."""

    def __init__(self, model_dir: str = "final_model_artifacts"):
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
        features : [OM_Percent, P_ppm, K_ppm, pH]
            Names must match ``final_model_artifacts`` (LightGBM / scaler training columns).

        Returns
        -------
        dict with keys: prediction, probabilities (all), top_3
        """
        soil_sample = pd.DataFrame(
            [
                {
                    "OM_Percent": features[0],
                    "P_ppm": features[1],
                    "K_ppm": features[2],
                    "pH": features[3],
                }
            ]
        )

        input_data = soil_sample

        out = np.asarray(self._model.predict(input_data))
        crops = self._label_encoder.classes_

        all_probs: List[Dict[str, Any]] = []
        top_3: List[Dict[str, Any]] = []
        raw_prediction: int
        probs_row: Optional[np.ndarray] = None

        # LightGBM Booster: ``predict`` returns (n_samples, n_classes) probabilities, no ``predict_proba``.
        if out.ndim == 2 and out.shape[1] > 1:
            probs_row = np.asarray(out[0], dtype=float)
            raw_prediction = int(np.argmax(probs_row))
        elif hasattr(self._model, "predict_proba"):
            flat = np.ravel(out)
            raw_prediction = int(flat[0])
            probs_row = np.asarray(
                self._model.predict_proba(input_data)[0], dtype=float
            )
        else:
            flat = np.ravel(out)
            raw_prediction = int(flat[0]) if flat.size == 1 else int(
                np.argmax(flat)
            )

        prediction_name = str(
            self._label_encoder.inverse_transform([raw_prediction])[0]
        )

        if probs_row is not None:
            crop_probs = sorted(
                zip(crops, probs_row), key=lambda x: x[1], reverse=True
            )
            all_probs = [
                {"crop_class": str(c), "probability": float(p)}
                for c, p in crop_probs
            ]
            top_3 = all_probs[:3]

        return {
            "prediction": prediction_name,
            "probabilities": all_probs,
            "top_3": top_3,
        }
