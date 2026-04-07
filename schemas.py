from typing import List, Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: List[float]
    farm_id: Optional[str] = None


class FertilizerPredictRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    crop: str
    farm_id: Optional[str] = None
