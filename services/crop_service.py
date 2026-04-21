import json
import joblib
import os
import warnings

import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Margin (in pH units) around the ideal range that counts as "marginal"
_PH_MARGIN = 0.5


class CropService:
    """Handles crop recommendation via the LightGBM model."""

    def __init__(self, model_dir: str = "final_model_artifacts", data_dir: str = "data"):
        self._model = joblib.load(os.path.join(model_dir, "lgbm_crop_model.pkl"))
        self._label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

        try:
            self._scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        except FileNotFoundError:
            self._scaler = None

        # BSWM Table 2 – Soil pH preferences per crop
        ph_path = os.path.join(data_dir, "crop_ph_preferences.json")
        try:
            with open(ph_path) as f:
                ph_data = json.load(f)
            self._ph_prefs: Dict[str, Dict] = ph_data.get("crops", {})
        except FileNotFoundError:
            self._ph_prefs = {}

    # ------------------------------------------------------------------
    # pH suitability annotation
    # ------------------------------------------------------------------
    def annotate_ph_suitability(
        self, crop_probs: List[Dict[str, Any]], soil_ph: float,
    ) -> List[Dict[str, Any]]:
        """Add ``ph_suitability`` info to each crop probability entry.

        Levels:
        - **suitable**  – soil pH is within the crop's preferred range
        - **marginal**  – soil pH is within 0.5 pH units outside the range
        - **unsuitable** – soil pH is clearly outside the preferred range
        - **unknown**    – no pH preference data for this crop

        Also adds ``ph_preferred_range`` (e.g. "4.5 – 6.5") and ``ph_category``.
        """
        annotated = []
        for entry in crop_probs:
            crop_name = entry.get("crop_class", "")
            pref = self._resolve_ph_pref(crop_name)

            item = dict(entry)
            if pref is None:
                item["ph_suitability"] = "unknown"
            else:
                ph_min = pref["ph_min"]
                ph_max = pref["ph_max"]
                item["ph_preferred_range"] = f"{ph_min} – {ph_max}"
                item["ph_category"] = pref.get("category", "")

                if ph_min <= soil_ph <= ph_max:
                    item["ph_suitability"] = "suitable"
                elif (ph_min - _PH_MARGIN) <= soil_ph <= (ph_max + _PH_MARGIN):
                    item["ph_suitability"] = "marginal"
                else:
                    item["ph_suitability"] = "unsuitable"
            annotated.append(item)
        return annotated

    def _resolve_ph_pref(self, crop_name: str) -> Optional[Dict]:
        """Case-insensitive lookup into the pH preferences map."""
        exact = self._ph_prefs.get(crop_name)
        if exact is not None:
            return exact
        crop_lower = crop_name.lower()
        for key, value in self._ph_prefs.items():
            if key.lower() == crop_lower:
                return value
        return None

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
        om  = features[0]
        p   = features[1]
        k   = features[2]
        ph  = features[3]
        eps = 1e-6  # guard against division by zero

        soil_sample = pd.DataFrame(
            [
                {
                    "OM_Percent": om,
                    "P_ppm": p,
                    "K_ppm": k,
                    "pH": ph,
                    "P_K_ratio": p / max(k, eps),
                    "OM_pH_interaction": om * ph,
                    "K_pH_interaction": k * ph,
                    "P_OM_ratio": p / max(om, eps),
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
