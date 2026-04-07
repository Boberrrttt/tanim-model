import json
import os

from typing import Any, Dict, List, Optional


class FertilizerService:
    """Rule-based fertilizer recommendation using BSWM tables.

    Flow: sensor NPK → classify L/M/H → per-crop rate lookup → bag calculation.
    """

    # Soil science thresholds for sensor-based NPK classification
    _NPK_THRESHOLDS = {
        "nitrogen":   {"low_ceil": 2.0,  "high_floor": 4.0},   # OM %
        "phosphorus": {"low_ceil": 10,   "high_floor": 25},     # ppm
        "potassium":  {"low_ceil": 60,   "high_floor": 120},    # ppm
    }

    # Per-crop mode-of-application instructions
    _MODE_OF_APPLICATION: Dict[str, Dict[str, str]] = {
        "Cassava": {
            "first_application": (
                "Apply the phosphorus and potassium fertilizers "
                "including \u00bd of nitrogen at planting."
            ),
            "second_application": (
                "Sidedress with remaining nitrogen fertilizer "
                "2-3 months after planting during hilling up."
            ),
            "organic_fertilizer": (
                "Apply 14 days to 1 month before planting."
            ),
        },
    }

    _DEFAULT_MODE: Dict[str, str] = {
        "first_application": (
            "Apply phosphorus and potassium fertilizers "
            "including \u00bd of nitrogen at planting."
        ),
        "second_application": (
            "Sidedress with remaining nitrogen fertilizer "
            "at the appropriate growth stage."
        ),
        "organic_fertilizer": (
            "Apply 14 days to 1 month before planting."
        ),
    }

    def __init__(self, data_dir: str = "data"):
        rates_path = os.path.join(data_dir, "crop_fertilizer_rates.json")
        with open(rates_path, "r") as f:
            all_rates: Dict = json.load(f)

        self._default_rates = all_rates.pop("_default", {
            "nitrogen":   {"Low": 90,  "Medium": 60,  "High": 30},
            "phosphorus": {"Low": 60,  "Medium": 35,  "High": 20},
            "potassium":  {"Low": 60,  "Medium": 45,  "High": 30},
        })
        self._crop_rates = all_rates

    # ------------------------------------------------------------------
    # NPK Classification
    # ------------------------------------------------------------------

    @classmethod
    def classify_npk(
        cls, nitrogen: float, phosphorus: float, potassium: float
    ) -> Dict[str, str]:
        """Classify quantitative NPK values into Low / Medium / High.

        Standard soil science thresholds:
            Nitrogen  (OM %):  Low < 2.0,   Medium 2.0–4.0,   High > 4.0
            Phosphorus (ppm):  Low < 10,    Medium 10–25,      High > 25
            Potassium  (ppm):  Low < 60,    Medium 60–120,     High > 120

        Low  = soil is deficient  → needs more fertilizer
        High = soil is sufficient → needs less fertilizer
        """
        def _level(value: float, low_ceil: float, high_floor: float) -> str:
            if value < low_ceil:
                return "Low"
            elif value <= high_floor:
                return "Medium"
            return "High"

        return {
            name: _level(val, **cls._NPK_THRESHOLDS[name])
            for name, val in [
                ("nitrogen", nitrogen),
                ("phosphorus", phosphorus),
                ("potassium", potassium),
            ]
        }

    # ------------------------------------------------------------------
    # Per-crop fertilizer rate lookup
    # ------------------------------------------------------------------

    def get_fertilizer_rate(
        self, npk_levels: Dict[str, str], crop: Optional[str] = None
    ) -> Dict[str, Any]:
        """Look up per-crop N-P₂O₅-K₂O rates (kg/ha) from L/M/H levels."""
        rates = self._resolve_crop_rates(crop)

        n_rate = rates["nitrogen"].get(npk_levels["nitrogen"], 0)
        p_rate = rates["phosphorus"].get(npk_levels["phosphorus"], 0)
        k_rate = rates["potassium"].get(npk_levels["potassium"], 0)

        return {
            "n": n_rate,
            "p": p_rate,
            "k": k_rate,
            "formatted": f"{n_rate} - {p_rate} - {k_rate}",
        }

    def _resolve_crop_rates(self, crop: Optional[str]) -> Dict:
        if crop:
            rates = self._crop_rates.get(crop)
            if rates is None:
                for key, value in self._crop_rates.items():
                    if key.lower() == crop.lower():
                        return value
            if rates is not None:
                return rates
        return self._default_rates

    # ------------------------------------------------------------------
    # Bag calculation (BSWM Standard)
    # ------------------------------------------------------------------
    #
    # Commercial fertilizers (50 kg bag):
    #   14-14-14 (Complete)            → 7 kg N,  7 kg P₂O₅, 7 kg K₂O / bag
    #   0-18-0   (Solophos / SSP)      → 0 kg N,  9 kg P₂O₅, 0 kg K₂O / bag
    #   16-20-0  (Ammonium Phosphate)  → 8 kg N, 10 kg P₂O₅, 0 kg K₂O / bag
    #   0-0-60   (Muriate of Potash)   → 0 kg N,  0 kg P₂O₅, 30 kg K₂O / bag
    #   46-0-0   (Urea)               → 23 kg N, 0 kg P₂O₅, 0 kg K₂O / bag
    #
    # Option 1: 14-14-14 + 0-18-0 + 46-0-0  (K drives 14-14-14)
    # Option 2: 16-20-0 + 0-0-60 + 46-0-0   (P drives 16-20-0)
    #
    # Both options split ALL fertilizers equally between 1st & 2nd application.
    # ------------------------------------------------------------------

    def calculate_application(
        self, n_rate: int, p_rate: int, k_rate: int
    ) -> Dict[str, Any]:
        """Convert NPK rates (kg/ha) into bags of commercial fertilizer."""
        return {
            "organic_fertilizer": "10 bags/ha",
            "option_1": self._option_1(n_rate, p_rate, k_rate),
            "option_2": self._option_2(n_rate, p_rate, k_rate),
        }

    # --- Option 1: 14-14-14 + 0-18-0 + 46-0-0 ---

    def _option_1(
        self, n_rate: int, p_rate: int, k_rate: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        total_complete = k_rate / 7 if k_rate > 0 else 0.0
        per_app_complete = self._round_quarter(total_complete / 2)

        p_from_complete = (per_app_complete * 2) * 7
        remaining_p = max(0, p_rate - p_from_complete)
        total_ssp = remaining_p / 9 if remaining_p > 0 else 0.0
        per_app_ssp = self._round_quarter(total_ssp / 2)

        n_from_complete = (per_app_complete * 2) * 7
        remaining_n = max(0, n_rate - n_from_complete)
        total_urea = remaining_n / 23 if remaining_n > 0 else 0.0
        per_app_urea = self._round_quarter(total_urea / 2)

        app = self._build_app_list([
            (per_app_complete, "Complete Fertilizer (14-14-14)"),
            (per_app_ssp, "Solophos (0-18-0)"),
            (per_app_urea, "Urea (46-0-0)"),
        ])
        return {"first_application": app, "second_application": app}

    # --- Option 2: 16-20-0 + 0-0-60 + 46-0-0 ---

    def _option_2(
        self, n_rate: int, p_rate: int, k_rate: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        total_ap = p_rate / 10 if p_rate > 0 else 0.0
        per_app_ap = self._round_quarter(total_ap / 2)

        total_mop = k_rate / 30 if k_rate > 0 else 0.0
        per_app_mop = self._round_quarter(total_mop / 2)

        n_from_ap = (per_app_ap * 2) * 8
        remaining_n = max(0, n_rate - n_from_ap)
        total_urea = remaining_n / 23 if remaining_n > 0 else 0.0
        per_app_urea = self._round_quarter(total_urea / 2)

        app = self._build_app_list([
            (per_app_ap, "Ammonium Phosphate (16-20-0)"),
            (per_app_mop, "Muriate of Potash (0-0-60)"),
            (per_app_urea, "Urea (46-0-0)"),
        ])
        return {"first_application": app, "second_application": app}

    # ------------------------------------------------------------------
    # Mode of application
    # ------------------------------------------------------------------

    def get_mode_of_application(self, crop: Optional[str] = None) -> Dict[str, str]:
        """Return crop-specific or default mode-of-application instructions."""
        if crop:
            mode = self._MODE_OF_APPLICATION.get(crop)
            if mode is None:
                for key, value in self._MODE_OF_APPLICATION.items():
                    if key.lower() == crop.lower():
                        return value
            if mode is not None:
                return mode
        return self._DEFAULT_MODE

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _round_quarter(x: float) -> float:
        """Round to the nearest 0.25."""
        return round(x * 4) / 4

    @staticmethod
    def _build_app_list(
        items: List[tuple],
    ) -> List[Dict[str, Any]]:
        return [
            {"bags_per_ha": bags, "fertilizer": name}
            for bags, name in items
            if bags > 0
        ]
