import copy
import json
import logging
import os

from fastapi import FastAPI, HTTPException

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from schemas import PredictRequest, FertilizerPredictRequest
from services.crop_service import CropService
from services.farming_timeline import build_farming_timeline
from services.fertilizer_service import FertilizerService
from services.api_sync import APISyncService
from services import pending_soil

app = FastAPI(title="ML Inference API")

logger = logging.getLogger("tanim_model")

crop_service = CropService()
fertilizer_service = FertilizerService()
api_sync = APISyncService()


def _defer_soil_sync() -> bool:
    """When true: skip APISyncService on /predict; ESP farm_id (or any farm_id) must not trigger API writes."""
    v = os.environ.get("DEFER_SOIL_SYNC", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _pending_soil_body_for_log(data: dict) -> dict:
    """Shrink long probability lists so logs stay readable."""
    trimmed = copy.deepcopy(data)
    probs = trimmed.get("probabilities")
    if isinstance(probs, list) and len(probs) > 5:
        trimmed["probabilities"] = probs[:5] + [
            {"_note": f"{len(probs) - 5} more crop_class entries omitted from log"}
        ]
    return trimmed


def _print_json(tag: str, obj: object) -> None:
    """Stdout debug print for request/response bodies."""
    try:
        if hasattr(obj, "model_dump"):
            data = obj.model_dump()
        elif isinstance(obj, dict):
            data = obj
        else:
            data = {"value": repr(obj)}
        print(f"{tag} {json.dumps(data, default=str)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{tag} <serialize error: {exc}>", flush=True)


def _print_pending_cache(tag: str) -> None:
    """Print global pending_soil slot (trimmed probabilities for terminal width)."""
    snap = pending_soil.get_snapshot()
    if snap is None:
        print(f"{tag} pending_soil cache: <empty>", flush=True)
        return
    trimmed = _pending_soil_body_for_log(snap)
    print(
        f"{tag} pending_soil cache: {json.dumps(trimmed, default=str)}",
        flush=True,
    )


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/pending/soil")
def get_pending_soil():
    """Return the latest cached /predict snapshot from this ML instance, or 404 if empty."""
    snap = pending_soil.get_snapshot()
    if snap is None:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "waiting",
                "message": "No cached reading on this ML service yet",
            },
        )
    body = {"status": "success", "data": snap}
    try:
        log_data = _pending_soil_body_for_log(snap)
        payload = json.dumps({"status": "success", "data": log_data}, default=str)
        logger.info("GET /pending/soil → 200 | %s", payload)
        # Always visible in the uvicorn terminal (custom loggers may be filtered).
        print(f"[ML GET /pending/soil] {payload}", flush=True)
    except Exception:
        logger.exception("GET /pending/soil logging failed")
    return body


@app.delete("/pending/soil")
def delete_pending_soil():
    """Clear the global pending slot (e.g. after a successful Save on tanim-api)."""
    cleared = pending_soil.clear()
    return {
        "status": "success",
        "message": "Pending cache cleared" if cleared else "Nothing to clear",
        "cleared": cleared,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """Predict the best crop for given soil features + NPK classification."""
    try:
        _print_json("[ML POST /predict] received body", request)
        features = request.features
        result = crop_service.predict(features)

        npk_levels = FertilizerService.classify_npk(
            nitrogen=features[0],
            phosphorus=features[1],
            potassium=features[2],
        )

        predict_data = {
            "prediction": result["prediction"],
            "npk_levels": npk_levels,
            "probabilities": result["top_3"],
        }
        predict_response = {
            "status": "success",
            "message": "Prediction successful",
            "data": predict_data,
        }
        if hasattr(request, "model_dump"):
            request_payload = request.model_dump(exclude_unset=True)
        else:
            request_payload = request.dict(exclude_unset=True)

        pending_soil.upsert(
            features=features,
            prediction=result["prediction"],
            npk_levels=npk_levels,
            probabilities=result["probabilities"],
            probabilities_top_3=result["top_3"],
            farm_id=request.farm_id,
            crop_model={
                "prediction": result["prediction"],
                "probabilities": result["probabilities"],
                "top_3": result["top_3"],
            },
            request_payload=request_payload,
            predict_response=predict_response,
        )
        _print_pending_cache("[ML POST /predict] after upsert")

        defer = _defer_soil_sync()
        if not defer and request.farm_id:
            api_sync.sync_soil_health_test(features, request.farm_id)
            api_sync.sync_crop_recommendation(
                request.farm_id, result["probabilities"]
            )

        _print_json("[ML POST /predict] response body", predict_response)
        return predict_response

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
        _print_json("[ML POST /predict/fertilizer] received body", request)
        _print_pending_cache("[ML POST /predict/fertilizer] current")

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

        farming_timeline = build_farming_timeline(
            request.crop, request.cycle_start_date
        )

        response_body = {
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
                "farming_timeline": farming_timeline,
            },
        }
        _print_json("[ML POST /predict/fertilizer] response body", response_body)
        _print_pending_cache("[ML POST /predict/fertilizer] pending_soil cache (same slot as /predict)")
        return response_body

    except Exception as e:
        return {
            "status": "error",
            "message": f"Fertilizer recommendation failed: {str(e)}",
        }
