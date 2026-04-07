import json
import os
import urllib.error
import urllib.request

from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse


class APISyncService:
    """Sync soil test results and crop recommendations to the main Tanim API."""

    def __init__(self):
        self._origin = self._resolve_origin()

    # Public API
    def sync_soil_health_test(self, features: List[float], farm_id: str) -> None:
        """POST/PUT a soil health test record to the main API."""
        url = f"{self._origin}/api/v1/test/"
        payload = {
            "nitrogen": float(features[0]),
            "phosphorus": float(features[1]),
            "potassium": float(features[2]),
            "ph": float(features[3]),
            "salinity": 0.0,
            "temperature": float(features[4]) if len(features) > 4 else 0.0,
            "moisture": float(features[5]) if len(features) > 5 else 0.0,
            "farm_id": farm_id,
        }
        self._upsert(url, payload)

    def sync_crop_recommendation(
        self, farm_id: str, probabilities: List[Dict[str, Any]]
    ) -> None:
        """POST/PUT crop recommendation probabilities to the main API."""
        url = f"{self._origin}/api/v1/crop-recommendations/"
        self._upsert(url, {"farm_id": farm_id, "probabilities": probabilities})

    # Internal helpers
    def _upsert(self, url: str, payload: Dict) -> None:
        """POST first; on 409 Conflict fall back to PUT."""
        body = json.dumps(payload).encode("utf-8")

        def _make_req(method: str) -> urllib.request.Request:
            return urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method=method,
            )

        try:
            with urllib.request.urlopen(_make_req("POST"), timeout=15) as resp:
                resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 409:
                try:
                    with urllib.request.urlopen(_make_req("PUT"), timeout=15) as resp:
                        resp.read()
                except (urllib.error.URLError, urllib.error.HTTPError,
                        TimeoutError, OSError) as e2:
                    print(f"API sync (PUT) failed ({url}): {e2}")
            else:
                print(f"API sync failed ({url}): {e}")
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as e:
            print(f"API sync failed ({url}): {e}")

    @staticmethod
    def _resolve_origin() -> str:
        """Resolve the origin URL, replacing 0.0.0.0 with 127.0.0.1."""
        raw = os.environ.get("TANIM_API_BASE_URL", "http://0.0.0.0:8000").rstrip("/")
        if raw.endswith("/api/v1"):
            raw = raw[: -len("/api/v1")].rstrip("/")
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        if host == "0.0.0.0":
            host = "127.0.0.1"
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        base = urlunparse((scheme, netloc, "", "", "", "")).rstrip("/")
        return base or "http://127.0.0.1:8000"
