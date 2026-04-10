from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """ESP / device ingest: **features** plus optional ``farm_id`` and coordinates for cache / sync."""

    features: List[float] = Field(
        ...,
        description="Sensor vector; first four used as OM_Percent, P_ppm, K_ppm, pH for the crop model (matches training feature names). Extra values map to temperature, moisture, EC for cached fertilizer_inputs.",
    )
    farm_id: Optional[str] = Field(
        default=None,
        description="Optional; not sent by ESP in deferred mode. When DEFER_SOIL_SYNC is off, triggers APISyncService if set.",
    )
    lat: Optional[float] = Field(
        default=None,
        description="Optional latitude; stored on pending cache as part of the received payload.",
    )
    lng: Optional[float] = Field(
        default=None,
        description="Optional longitude; stored on pending cache as part of the received payload.",
    )


class FertilizerPredictRequest(BaseModel):
    """BSWM rule-based recommendation: N, P, K, pH, and crop from the client (e.g. mobile app)."""

    model_config = {"extra": "ignore"}

    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    crop: str
    farm_id: Optional[str] = None
    cycle_start_date: Optional[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD; echoed in farming_timeline when set (e.g. soil reading date).",
    )
