"""Single-slot pending soil + crop snapshot on the ML service (global latest /predict).

Each snapshot includes:
- **Inputs**: raw ``features``, derived ``soil`` / ``fertilizer_inputs`` scalars.
- **Model output**: ``prediction``, full ranked ``probabilities`` (all classes), ``probabilities_top_3``,
  and ``crop_model`` (verbatim ``CropService.predict`` return: prediction, probabilities, top_3).
- **NPK classification**: ``npk_levels`` from BSWM-style L/M/H rules.
- **Optional** ``farm_id`` when the client sent it on ``POST /predict``.
- **``request_payload``**: parsed ``POST /predict`` body (e.g. ``features``, ``farm_id``, ``lat``, ``lng``).
- **``predict_response``**: full successful JSON body returned by ``POST /predict`` (``status``, ``message``, ``data``).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_store: Optional[Dict[str, Any]] = None
_sequence = 0


def _soil_scalars_from_features(features: List[float]) -> Dict[str, float]:
    """Align with APISyncService / soil health test shape (indices may extend for temp/moisture/ec)."""
    return {
        "nitrogen": float(features[0]) if len(features) > 0 else 0.0,
        "phosphorus": float(features[1]) if len(features) > 1 else 0.0,
        "potassium": float(features[2]) if len(features) > 2 else 0.0,
        "ph": float(features[3]) if len(features) > 3 else 0.0,
        "salinity": 0.0,
        "temperature": float(features[4]) if len(features) > 4 else 0.0,
        "moisture": float(features[5]) if len(features) > 5 else 0.0,
    }


def _fertilizer_inputs_from_features(features: List[float]) -> Dict[str, float]:
    soil = _soil_scalars_from_features(features)
    ec = float(features[6]) if len(features) > 6 else 0.0
    return {
        "nitrogen": soil["nitrogen"],
        "phosphorus": soil["phosphorus"],
        "potassium": soil["potassium"],
        "ph": soil["ph"],
        "temperature": soil["temperature"],
        "ec": ec,
        "moisture": soil["moisture"],
    }


def upsert(
    features: List[float],
    prediction: str,
    npk_levels: Dict[str, Any],
    probabilities: List[Dict[str, Any]],
    probabilities_top_3: List[Dict[str, Any]],
    *,
    farm_id: Optional[str] = None,
    crop_model: Optional[Dict[str, Any]] = None,
    request_payload: Optional[Dict[str, Any]] = None,
    predict_response: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store latest snapshot; returns the stored document."""
    global _store, _sequence
    with _lock:
        _sequence += 1
        now = datetime.now(timezone.utc)
        model_block = crop_model or {
            "prediction": prediction,
            "probabilities": probabilities,
            "top_3": probabilities_top_3,
        }
        doc: Dict[str, Any] = {
            "received_at": now.isoformat(),
            "sequence": _sequence,
            "features": [float(x) for x in features],
            "prediction": prediction,
            "npk_levels": npk_levels,
            "probabilities": probabilities,
            "probabilities_top_3": probabilities_top_3,
            "crop_model": model_block,
            "soil": _soil_scalars_from_features(features),
            "fertilizer_inputs": _fertilizer_inputs_from_features(features),
        }
        if farm_id is not None and str(farm_id).strip() != "":
            doc["farm_id"] = str(farm_id).strip()
        if request_payload is not None:
            doc["request_payload"] = request_payload
        if predict_response is not None:
            doc["predict_response"] = predict_response
        _store = doc
        return doc.copy()


def get_snapshot() -> Optional[Dict[str, Any]]:
    with _lock:
        if _store is None:
            return None
        return _store.copy()


def clear() -> bool:
    """Return True if something was cleared."""
    global _store
    with _lock:
        if _store is None:
            return False
        _store = None
        return True
