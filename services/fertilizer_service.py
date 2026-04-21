import json
import os

from typing import Any, Dict, List, Optional


class FertilizerService:
    """Rule-based fertilizer recommendation using BSWM tables.

    Flow: sensor NPK → classify L/M/H → per-crop rate lookup → bag calculation.

    All static data (crop rates, mode-of-application instructions, split rules)
    is loaded from JSON files in the ``data/`` directory so this module stays
    focused on the calculation logic.
    """

    # ------------------------------------------------------------------
    # NPK classification thresholds (BSWM soil-analysis standard)
    # ------------------------------------------------------------------
    #   N  – Organic Matter %   :  Low 0–1.7,   Medium >1.7–3.0,  High >3.0
    #   P  – Phosphorus (Bray 1):  Low 0–10,    Medium >10–20,    High >20
    #        Phosphorus (Olsen) :  Low 0–7,     Medium >7–25,     High >25
    #   K  – Potassium ppm      :  Low 0–117,   Medium >117–235,  High >235
    # ------------------------------------------------------------------
    _NPK_THRESHOLDS = {
        "nitrogen":   {"low_ceil": 1.7,  "high_floor": 3.0},   # OM %
        "phosphorus_bray1": {"low_ceil": 10,   "high_floor": 20},    # ppm  (Bray 1)
        "phosphorus_olsen": {"low_ceil": 7,    "high_floor": 25},    # ppm  (Olsen)
        "potassium":  {"low_ceil": 117,  "high_floor": 235},   # ppm
    }

    # ------------------------------------------------------------------
    # Commercial fertilizer specs  (50 kg bag)
    # ------------------------------------------------------------------
    _FERTILIZERS: Dict[str, Dict[str, Any]] = {
        "complete":  {"grade": "14-14-14", "label": "Complete Fertilizer (14-14-14)", "n": 7,  "p": 7,  "k": 7},
        "ssp":       {"grade": "0-18-0",   "label": "Solophos (0-18-0)",             "n": 0,  "p": 9,  "k": 0},
        "ap":        {"grade": "16-20-0",  "label": "Ammonium Phosphate (16-20-0)",  "n": 8,  "p": 10, "k": 0},
        "mop":       {"grade": "0-0-60",   "label": "Muriate of Potash (0-0-60)",    "n": 0,  "p": 0,  "k": 30},
        "urea":      {"grade": "46-0-0",   "label": "Urea (46-0-0)",                "n": 23, "p": 0,  "k": 0},
    }

    def __init__(self, data_dir: str = "data"):
        # Crop fertilizer rates (kg/ha per L/M/H)
        with open(os.path.join(data_dir, "crop_fertilizer_rates.json")) as f:
            all_rates: Dict = json.load(f)
        self._default_rates = all_rates.pop("_default", {
            "nitrogen":   {"Low": 90,  "Medium": 60,  "High": 30},
            "phosphorus": {"Low": 60,  "Medium": 35,  "High": 20},
            "potassium":  {"Low": 60,  "Medium": 45,  "High": 30},
        })
        self._crop_rates = all_rates

        # Mode-of-application instructions & split rules
        with open(os.path.join(data_dir, "mode_of_application.json")) as f:
            moa: Dict = json.load(f)
        self._crop_modes: Dict[str, Dict] = moa.get("crops", {})
        self._default_mode: Dict[str, str] = moa.get("_default_mode", {})
        self._split_rules: Dict[str, List[List[float]]] = moa.get("_split_rules", {})
        self._default_split: List[List[float]] = moa.get("_default_split", [[0.5, 1.0, 1.0], [0.5, 0.0, 0.0]])

    # ------------------------------------------------------------------
    # NPK classification
    # ------------------------------------------------------------------
    @classmethod
    def classify_npk(
        cls, nitrogen: float, phosphorus: float, potassium: float, ph: Optional[float] = None
    ) -> Dict[str, str]:
        """Classify sensor NPK values into Low / Medium / High.

        Boundary values are inclusive on the low side (≤ low_ceil → Low).
        Uses Bray-1 test thresholds for pH < 7.3, and Olsen test for pH >= 7.3.
        Defaults to Bray-1 if pH is not provided.
        """
        def _level(value: float, low_ceil: float, high_floor: float) -> str:
            if value <= low_ceil:
                return "Low"
            if value <= high_floor:
                return "Medium"
            return "High"

        p_key = "phosphorus_bray1"
        if ph is not None and ph >= 7.3:
            p_key = "phosphorus_olsen"

        return {
            "nitrogen": _level(nitrogen, **cls._NPK_THRESHOLDS["nitrogen"]),
            "phosphorus": _level(phosphorus, **cls._NPK_THRESHOLDS[p_key]),
            "potassium": _level(potassium, **cls._NPK_THRESHOLDS["potassium"]),
        }

    # ------------------------------------------------------------------
    # Fertilizer rate lookup
    # ------------------------------------------------------------------
    def get_fertilizer_rate(
        self, npk_levels: Dict[str, str], crop: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up per-crop N-P₂O₅-K₂O rates (kg/ha) from L/M/H levels."""
        rates = self._resolve(self._crop_rates, crop, self._default_rates)
        n = rates["nitrogen"].get(npk_levels["nitrogen"], 0)
        p = rates["phosphorus"].get(npk_levels["phosphorus"], 0)
        k = rates["potassium"].get(npk_levels["potassium"], 0)
        return {"n": n, "p": p, "k": k, "formatted": f"{n} - {p} - {k}"}

    # ------------------------------------------------------------------
    # Mode of application
    # ------------------------------------------------------------------
    def get_mode_of_application(self, crop: Optional[str] = None) -> Dict[str, str]:
        """Return crop-specific or default mode-of-application instructions."""
        return self._resolve(self._crop_modes, crop, self._default_mode)

    # ------------------------------------------------------------------
    # Bag calculation  (BSWM standard, two fertilizer-mix options)
    # ------------------------------------------------------------------
    def calculate_application(
        self, n_rate: int, p_rate: int, k_rate: int,
        crop: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert NPK rates (kg/ha) into bags of commercial fertilizer."""
        splits = self._split_rates(n_rate, p_rate, k_rate, crop)
        return {
            "organic_fertilizer": "10 bags/ha",
            "option_1": self._option_bags(splits, self._option1_sequence),
            "option_2": self._option_bags(splits, self._option2_sequence),
        }

    # --- Option helpers (each returns [(bags, label), …] for one window) ---

    def _option1_sequence(self, n: float, p: float, k: float) -> List[tuple]:
        """14-14-14 + 0-18-0 + 46-0-0  (K drives Complete bags)."""
        f = self._FERTILIZERS
        bags_complete = self._q(k / f["complete"]["k"]) if k > 0 else 0.0
        remaining_p = max(0, p - bags_complete * f["complete"]["p"])
        remaining_n = max(0, n - bags_complete * f["complete"]["n"])
        bags_ssp  = self._q(remaining_p / f["ssp"]["p"])  if remaining_p > 0 else 0.0
        bags_urea = self._q(remaining_n / f["urea"]["n"]) if remaining_n > 0 else 0.0
        return [
            (bags_complete, f["complete"]["label"]),
            (bags_ssp,      f["ssp"]["label"]),
            (bags_urea,     f["urea"]["label"]),
        ]

    def _option2_sequence(self, n: float, p: float, k: float) -> List[tuple]:
        """16-20-0 + 0-0-60 + 46-0-0  (P drives AP bags)."""
        f = self._FERTILIZERS
        bags_ap  = self._q(p / f["ap"]["p"])  if p > 0 else 0.0
        bags_mop = self._q(k / f["mop"]["k"]) if k > 0 else 0.0
        remaining_n = max(0, n - bags_ap * f["ap"]["n"])
        bags_urea = self._q(remaining_n / f["urea"]["n"]) if remaining_n > 0 else 0.0
        return [
            (bags_ap,   f["ap"]["label"]),
            (bags_mop,  f["mop"]["label"]),
            (bags_urea, f["urea"]["label"]),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_rates(
        self, n: float, p: float, k: float, crop: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """Per-application NPK kg/ha using the crop's split formula."""
        mode = self.get_mode_of_application(crop)
        formula = mode.get("split_formula", "")
        fractions = self._split_rules.get(formula, self._default_split)
        return [
            {"n": n * nf, "p": p * pf, "k": k * kf}
            for nf, pf, kf in fractions
        ]

    def _option_bags(
        self, splits: List[Dict[str, float]], calc_fn,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run *calc_fn* for each application window and aggregate results."""
        names = ["first_application", "second_application", "third_application"]
        return {
            names[i]: [
                {"bags_per_ha": bags, "fertilizer": label}
                for bags, label in calc_fn(s["n"], s["p"], s["k"])
                if bags > 0
            ]
            for i, s in enumerate(splits)
        }

    @staticmethod
    def _resolve(mapping: Dict, crop: Optional[str], default: Dict) -> Dict:
        """Case-insensitive crop lookup with fallback to *default*."""
        if crop:
            exact = mapping.get(crop)
            if exact is not None:
                return exact
            crop_lower = crop.lower()
            for key, value in mapping.items():
                if key.lower() == crop_lower:
                    return value
        return default

    @staticmethod
    def _q(x: float) -> float:
        """Round to nearest 0.25 (quarter-bag)."""
        return round(x * 4) / 4
