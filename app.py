from fastapi import FastAPI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schemas import PredictRequest, FertilizerPredictRequest
from services.crop_service import CropService
from services.fertilizer_service import FertilizerService
from services.api_sync import APISyncService

# ---------------------------------------------------------------------------
# Initialize services
# ---------------------------------------------------------------------------
app = FastAPI(title="ML Inference API")

crop_service = CropService()
fertilizer_service = FertilizerService()
api_sync = APISyncService()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    """Predict the best crop for given soil features + NPK classification."""
    try:
        features = request.features
        result = crop_service.predict(features)

        npk_levels = FertilizerService.classify_npk(
            nitrogen=features[0],
            phosphorus=features[1],
            potassium=features[2],
        )

        # --- Sync to main API (fire-and-forget) ---
        if request.farm_id:
            api_sync.sync_soil_health_test(features, request.farm_id)
            api_sync.sync_crop_recommendation(
                request.farm_id, result["probabilities"]
            )

        return {
            "status": "success",
            "message": "Prediction successful",
            "data": {
                "prediction": result["prediction"],
                "npk_levels": npk_levels,
                "probabilities": result["top_3"],
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
        }


@app.post("/predict/fertilizer")
def fertilizer(request: FertilizerPredictRequest):
    """Rule-based fertilizer recommendation using BSWM tables.

    Flow: sensor NPK → classify L/M/H → per-crop rate lookup → bag calculation.
    """
    try:
        # 1. Classify sensor readings into Low / Medium / High
        npk_levels = FertilizerService.classify_npk(
            nitrogen=request.nitrogen,
            phosphorus=request.phosphorus,
            potassium=request.potassium,
        )

        # 2. Look up per-crop fertilizer rates (kg/ha)
        rate = fertilizer_service.get_fertilizer_rate(npk_levels, crop=request.crop)

        # 3. Calculate commercial fertilizer bags
        application = fertilizer_service.calculate_application(
            rate["n"], rate["p"], rate["k"]
        )

        # 4. Get mode-of-application instructions
        mode = fertilizer_service.get_mode_of_application(request.crop)

        return {
            "status": "success",
            "message": "Fertilizer recommendation successful",
            "data": {
                "crop": request.crop,
                "soil_ph": request.ph,
                "nitrogen": npk_levels["nitrogen"],
                "phosphorus": npk_levels["phosphorus"],
                "potassium": npk_levels["potassium"],
                "fertilizer_recommendation_rate": rate["formatted"],
                "organic_fertilizer": application["organic_fertilizer"],
                "option_1": application["option_1"],
                "option_2": application["option_2"],
                "mode_of_application": mode,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Fertilizer recommendation failed: {str(e)}",
        }
