"""Crop growth timelines for `/predict/fertilizer` — mirrors `tanim-app/constants/crop-cycle*.ts`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

PHASE_NAMES: Tuple[str, ...] = ("Sowing", "Vegetative", "Flowering", "Harvest")


def _four_phase(
    total_days: int,
    spans: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    descriptions: Tuple[str, str, str, str],
    planting_window_note: Optional[str] = None,
) -> Dict[str, Any]:
    phases: List[Dict[str, Any]] = [
        {
            "name": PHASE_NAMES[i],
            "day_start": spans[i][0],
            "day_end": spans[i][1],
            "description": descriptions[i],
        }
        for i in range(4)
    ]
    out: Dict[str, Any] = {"total_days": total_days, "phases": phases}
    if planting_window_note:
        out["planting_window_note"] = planting_window_note
    return out


# Same numbers and copy as `TIMELINE_TEMPLATES` in tanim-app `constants/crop-cycle.ts`.
TIMELINE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "DEFAULT": _four_phase(
        85,
        ((1, 12), (13, 38), (39, 58), (59, 85)),
        (
            "Establishment from seed or transplant.",
            "Leaf and canopy development.",
            "Flowering and early fruit or grain set.",
            "Sizing, ripening, and harvest readiness.",
        ),
        "Generic template — match days-to-maturity on your seed or transplant label.",
    ),
    "CEREAL": _four_phase(
        105,
        ((1, 14), (15, 58), (59, 78), (79, 105)),
        (
            "Emergence and early vegetative growth.",
            "Stalk and leaf development until near reproductive stage.",
            "Pollination and reproductive window (e.g. tassel/silk for maize).",
            "Grain fill through physiological maturity / harvest moisture.",
        ),
        "Sweet corn is often shorter (~60–100 d); field/grain hybrids often 100–120+ d — check RM on bag.",
    ),
    "RICE": _four_phase(
        120,
        ((1, 30), (31, 70), (71, 100), (101, 120)),
        (
            "Establishment, rooting, and tillering.",
            "Active tillering and vegetative growth.",
            "Panicle development through heading and flowering.",
            "Ripening, grain fill, and harvest timing.",
        ),
        "IRRI: short types ~100–120 d, medium ~120–140 d, long 160+ — adjust for cultivar and transplant vs DSR.",
    ),
    "WHEAT": _four_phase(
        115,
        ((1, 15), (16, 65), (66, 95), (96, 115)),
        (
            "Emergence and early tillering.",
            "Tillering, stem elongation, and boot stage.",
            "Heading, anthesis, and early grain fill.",
            "Dough, maturity, and harvest readiness.",
        ),
        "Spring wheat commonly ~90–140 d from planting; variety maturity class matters.",
    ),
    "TOMATO": _four_phase(
        75,
        ((1, 7), (8, 28), (29, 50), (51, 75)),
        (
            "Transplant establishment.",
            "Vine and leaf growth.",
            "Bloom and fruit set.",
            "Fruit development, sizing, and harvest picks.",
        ),
        "Packet “days to maturity” is usually from transplant; early types ~65 d, late ~85 d.",
    ),
    "EGGPLANT": _four_phase(
        72,
        ((1, 12), (13, 32), (33, 48), (49, 72)),
        (
            "Transplant establishment and rooting.",
            "Branching and canopy build.",
            "Bloom and fruit set.",
            "Fruit enlargement and repeated harvest.",
        ),
        "Cultivars range ~50–80 d after transplant; warm nights help fruit set.",
    ),
    "POTATO": _four_phase(
        100,
        ((1, 20), (21, 50), (51, 70), (71, 100)),
        (
            "Emergence and stolon setup.",
            "Canopy closure and vegetative growth.",
            "Tuber initiation and early bulking.",
            "Tuber bulking, senescence, skin set, and harvest.",
        ),
        "Early varieties ~75–90 d; maincrop often 90–120 d — check your variety class.",
    ),
    "CABBAGE": _four_phase(
        68,
        ((1, 12), (13, 35), (36, 55), (56, 68)),
        (
            "Transplant establishment.",
            "Leaf frame before head.",
            "Head formation and firming.",
            "Maturation and harvest when head is firm.",
        ),
        "Fast cultivars ~55–65 d from transplant; storage types run longer.",
    ),
    "COTTON": _four_phase(
        165,
        ((1, 35), (36, 75), (76, 120), (121, 165)),
        (
            "Emergence through early vegetative nodes.",
            "Square formation and vegetative peak.",
            "Bloom, boll set, and early boll development.",
            "Boll fill, open boll, and defoliation/harvest timing.",
        ),
        "~150–180 d planting to harvest-ready; heat units (DD60s) often matter more than calendar days.",
    ),
    "TOBACCO": _four_phase(
        110,
        ((1, 20), (21, 55), (56, 80), (81, 110)),
        (
            "Rooting after transplant.",
            "Rapid leaf expansion (grand growth).",
            "Topping, suckering, and ripening prep.",
            "Leaf ripening, harvest, and curing window.",
        ),
        "FAO cites ~90–120 d frost-free after transplant; follow local rules and varieties.",
    ),
    "SUGARCANE": _four_phase(
        450,
        ((1, 60), (61, 300), (301, 390), (391, 450)),
        (
            "Shoot emergence, establishment, and early tillers.",
            "Tillering through grand growth and stalk elongation.",
            "Ripening phase — sucrose accumulation (UI label “Flowering” is shorthand; flowering is usually avoided).",
            "Harvest window for plant crop (~12–18 mo typical; region-dependent).",
        ),
        "FAO / regional guides: plant crop often 12–18 months (many areas ~15–16 mo optimum age).",
    ),
    "CUCURBIT": _four_phase(
        70,
        ((1, 10), (11, 35), (36, 55), (56, 70)),
        (
            "Emergence and vine/runner establishment.",
            "Vegetative growth and canopy development.",
            "Bloom and fruit set.",
            "Fruit sizing and harvest.",
        ),
        "Most cucurbits ~50–70 d to first harvest depending on variety and climate.",
    ),
    "LEAFY": _four_phase(
        45,
        ((1, 8), (9, 22), (23, 32), (33, 45)),
        (
            "Germination or transplant establishment.",
            "Rapid leaf production.",
            "Late vegetative / pre-bolt (if applicable).",
            "Harvest window for leaves or young shoots.",
        ),
        "Fast crops — timing varies sharply with heat, day length, and cultivar.",
    ),
    "LEGUME": _four_phase(
        75,
        ((1, 10), (11, 40), (41, 58), (59, 75)),
        (
            "Emergence and nodulation establishment.",
            "Vegetative growth and canopy.",
            "Flowering and pod or nut formation.",
            "Grain/pod fill and harvest.",
        ),
        "Compromise template: bush beans faster, peanuts longer — verify for your legume.",
    ),
    "OKRA": _four_phase(
        60,
        ((1, 8), (9, 30), (31, 45), (46, 60)),
        (
            "Emergence and early growth.",
            "Vegetative growth in warm conditions.",
            "Flowering and pod set.",
            "Repeated pod harvest while tender.",
        ),
        "Many cultivars ~50–60 d from planting in warm weather.",
    ),
    "ROOT_BULB": _four_phase(
        85,
        ((1, 12), (13, 45), (46, 65), (66, 85)),
        (
            "Emergence and root establishment.",
            "Vegetative tops and root enlargement.",
            "Bulking and maturity indicators.",
            "Harvest when size and quality targets are met.",
        ),
        "Onions and long-season roots may exceed this — use variety guidance.",
    ),
    "GINGER": _four_phase(
        270,
        ((1, 45), (46, 150), (151, 220), (221, 270)),
        (
            "Rhizome sprouting and shoot establishment.",
            "Strong vegetative growth.",
            "Rhizome expansion and maturation.",
            "Senescence cues and harvest of mature rhizomes.",
        ),
        "Often ~8–10 months to mature rhizome in the tropics.",
    ),
    "CAMOTE": _four_phase(
        120,
        ((1, 20), (21, 55), (56, 90), (91, 120)),
        (
            "Slip/root establishment and vine growth.",
            "Canopy development.",
            "Tuber initiation and early bulking.",
            "Tuber bulking and harvest before heavy frost.",
        ),
        "Sweet potato commonly ~100–120 d; taro (gabi) often longer in the field — illustrative here.",
    ),
    "CASSAVA": _four_phase(
        300,
        ((1, 30), (31, 120), (121, 240), (241, 300)),
        (
            "Establishment and early vegetative growth.",
            "Strong vegetative growth and starch accumulation start.",
            "Storage root bulking.",
            "Harvest window — often ~9–12 mo for many systems; wider range possible.",
        ),
        "Harvest age strongly affects yield and starch — follow local variety recommendations.",
    ),
    "STRAWBERRY": _four_phase(
        90,
        ((1, 14), (15, 45), (46, 70), (71, 90)),
        (
            "Planting and crown establishment.",
            "Runner and leaf development.",
            "Flowering and fruit set (system-dependent).",
            "Harvest period — highly dependent on June-bearing vs day-neutral systems.",
        ),
        "June-bearing types often little fruit in year one; day-neutral may fruit sooner — template is approximate.",
    ),
    "PINEAPPLE": _four_phase(
        540,
        ((1, 120), (121, 300), (301, 480), (481, 540)),
        (
            "Establishment from slips, suckers, or crowns.",
            "Vegetative growth and plant sizing.",
            "Induction/flowering and fruit development (region and practice dependent).",
            "Fruit maturation and harvest (~18 mo lower end; often longer).",
        ),
        "First harvest commonly ~18–24+ months from planting depending on propagation and induction.",
    ),
    "ADLAI": _four_phase(
        120,
        ((1, 25), (26, 70), (71, 100), (101, 120)),
        (
            "Emergence and early tillering.",
            "Vegetative growth.",
            "Reproductive development and grain fill.",
            "Maturity and harvest.",
        ),
        "Philippine extension often cites ~120 days or ~4–5 months; adjust for cultivar.",
    ),
    "FORAGE_GRASS": _four_phase(
        120,
        ((1, 30), (31, 75), (76, 100), (101, 120)),
        (
            "Establishment after planting or ratoon.",
            "Vegetative regrowth and tillering.",
            "Pre-cut vegetative peak.",
            "Cut-and-carry or grazing cycle end — management-driven, not fixed biology.",
        ),
        "Illustrative regrowth cycle; actual rotation depends on grazing or cutting schedule.",
    ),
    "PERENNIAL_YEAR1": _four_phase(
        365,
        ((1, 90), (91, 210), (211, 320), (321, 365)),
        (
            "Planting and establishment.",
            "Vegetative framework and root system building.",
            "Pre-productive growth or first reproductive cycle (species-dependent).",
            "Late establishment year — not a real “harvest” calendar for all trees.",
        ),
        "ILLUSTRATIVE ONLY: trees and plantation crops differ widely — use species-specific local guidance.",
    ),
}


def normalize_crop_label(raw: str) -> str:
    s = " ".join(raw.strip().lower().split())
    if s == "coonut":
        s = "coconut"
    if s == "ampalya":
        s = "ampalaya"
    if s == "radush":
        s = "radish"
    return s


# Normalized label → template id (from `CROP_LABEL_TO_TEMPLATE` in tanim-app `crop-cycle-aliases.ts`).
_ALIAS_ENTRIES: Tuple[Tuple[str, str], ...] = (
    ("algae", "DEFAULT"),
    ("banguhan", "DEFAULT"),
    ("buckwheat", "DEFAULT"),
    ("celery", "DEFAULT"),
    ("flowering plants", "DEFAULT"),
    ("herbs", "DEFAULT"),
    ("test data", "DEFAULT"),
    ("vegetables", "DEFAULT"),
    ("wild daisy", "DEFAULT"),
    ("corn", "CEREAL"),
    ("maize", "CEREAL"),
    ("millet", "CEREAL"),
    ("sorghum", "CEREAL"),
    ("rice (lowland)", "RICE"),
    ("rice (upland)", "RICE"),
    ("rice", "RICE"),
    ("wheat", "WHEAT"),
    ("cotton", "COTTON"),
    ("tomato", "TOMATO"),
    ("bell pepper", "TOMATO"),
    ("chili pepper", "TOMATO"),
    ("sweet pepper", "TOMATO"),
    ("eggplant", "EGGPLANT"),
    ("potato", "POTATO"),
    ("broccoli", "CABBAGE"),
    ("cabbage", "CABBAGE"),
    ("cabbage (chinese)", "CABBAGE"),
    ("tobacco", "TOBACCO"),
    ("sugarcane", "SUGARCANE"),
    ("ampalaya", "CUCURBIT"),
    ("chayote", "CUCURBIT"),
    ("cucumber", "CUCURBIT"),
    ("muskmelon (cantaloupe)", "CUCURBIT"),
    ("patola", "CUCURBIT"),
    ("squash", "CUCURBIT"),
    ("watermelon", "CUCURBIT"),
    ("alugbati", "LEAFY"),
    ("basil", "LEAFY"),
    ("kangkong", "LEAFY"),
    ("lettuce", "LEAFY"),
    ("baguio beans", "LEGUME"),
    ("beans (soybean)", "LEGUME"),
    ("beans (soybeans)", "LEGUME"),
    ("beans (string beans)", "LEGUME"),
    ("beans (stringbeans)", "LEGUME"),
    ("cowpea", "LEGUME"),
    ("mongo (mungbean)", "LEGUME"),
    ("peanut", "LEGUME"),
    ("peas", "LEGUME"),
    ("sitao", "LEGUME"),
    ("okra", "OKRA"),
    ("carrots", "ROOT_BULB"),
    ("onion", "ROOT_BULB"),
    ("radish", "ROOT_BULB"),
    ("ginger", "GINGER"),
    ("camote (sweet potato)", "CAMOTE"),
    ("gabi", "CAMOTE"),
    ("cassava", "CASSAVA"),
    ("strawberry", "STRAWBERRY"),
    ("pineapple", "PINEAPPLE"),
    ("adlai", "ADLAI"),
    ("alfalfa", "FORAGE_GRASS"),
    ("arachis pintoi", "FORAGE_GRASS"),
    ("gatton", "FORAGE_GRASS"),
    ("grass (pasture)", "FORAGE_GRASS"),
    ("green panic", "FORAGE_GRASS"),
    ("mombasa (forage)", "FORAGE_GRASS"),
    ("mombasa grass", "FORAGE_GRASS"),
    ("mulato grass", "FORAGE_GRASS"),
    ("napier", "FORAGE_GRASS"),
    ("abaca", "PERENNIAL_YEAR1"),
    ("acacia", "PERENNIAL_YEAR1"),
    ("avocado", "PERENNIAL_YEAR1"),
    ("bamboo", "PERENNIAL_YEAR1"),
    ("banana", "PERENNIAL_YEAR1"),
    ("caimito (star apple)", "PERENNIAL_YEAR1"),
    ("cacao", "PERENNIAL_YEAR1"),
    ("calamansi", "PERENNIAL_YEAR1"),
    ("coconut", "PERENNIAL_YEAR1"),
    ("coffee", "PERENNIAL_YEAR1"),
    ("dragonfruit", "PERENNIAL_YEAR1"),
    ("durian", "PERENNIAL_YEAR1"),
    ("falcata", "PERENNIAL_YEAR1"),
    ("forest trees", "PERENNIAL_YEAR1"),
    ("gmelina", "PERENNIAL_YEAR1"),
    ("golden tree", "PERENNIAL_YEAR1"),
    ("grapes", "PERENNIAL_YEAR1"),
    ("guyabano", "PERENNIAL_YEAR1"),
    ("jackfruit", "PERENNIAL_YEAR1"),
    ("lanzones", "PERENNIAL_YEAR1"),
    ("mahogany", "PERENNIAL_YEAR1"),
    ("mango", "PERENNIAL_YEAR1"),
    ("mangrove", "PERENNIAL_YEAR1"),
    ("mulberry", "PERENNIAL_YEAR1"),
    ("ornamental", "PERENNIAL_YEAR1"),
    ("palm", "PERENNIAL_YEAR1"),
    ("papaya", "PERENNIAL_YEAR1"),
    ("pine tree", "PERENNIAL_YEAR1"),
    ("rambutan", "PERENNIAL_YEAR1"),
    ("river tamarind", "PERENNIAL_YEAR1"),
    ("rubber", "PERENNIAL_YEAR1"),
    ("tea", "PERENNIAL_YEAR1"),
)

CROP_LABEL_TO_TEMPLATE: Dict[str, str] = {k: v for k, v in _ALIAS_ENTRIES}


def resolve_template_id(crop_name: str) -> str:
    key = normalize_crop_label(crop_name)
    return CROP_LABEL_TO_TEMPLATE.get(key, "DEFAULT")


def build_farming_timeline(
    crop: str, cycle_start_date: Optional[str] = None
) -> Dict[str, Any]:
    """Return timeline metadata for JSON `data.farming_timeline`."""
    template_id = resolve_template_id(crop)
    meta = TIMELINE_TEMPLATES[template_id]
    out: Dict[str, Any] = {
        "template_id": template_id,
        "total_days": meta["total_days"],
        "phases": meta["phases"],
    }
    if meta.get("planting_window_note"):
        out["planting_window_note"] = meta["planting_window_note"]
    if cycle_start_date:
        out["cycle_start_date"] = cycle_start_date.strip()
    return out
