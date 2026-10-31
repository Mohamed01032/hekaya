"""Personal route planner: pick the best artifacts for the visitor's time and interests."""
from __future__ import annotations

from typing import Any

ZONE_CHANGE_MIN = 5          # walking time between zones
MUST_SEE = {"tut-mask": 3.0, "ramesses-colossus": 2.0, "khufu-boat": 1.5, "grand-staircase": 1.0}


def plan_tour(kb, minutes: int, interests: list[str], audience: str = "adult",
              crowded: list[str] | None = None) -> dict[str, Any]:
    """Greedy knapsack by (interest score / minutes), then ordered along the museum's natural flow.

    `crowded` lists artifact ids currently busy (from live crowd data); they are de-prioritised.
    """
    minutes = max(15, min(int(minutes), 480))
    interests = [i for i in interests if i]
    crowded = set(crowded or [])
    speed = 1.25 if audience == "kids" else (1.0 if audience == "adult" else 0.8)

    def score(a: dict) -> float:
        s = 1.0 + MUST_SEE.get(a["id"], 0.0)
        s += 2.0 * len(set(a["tags"]) & set(interests))
        if a["id"] in crowded:
            s *= 0.5
        return s

    def cost(a: dict) -> float:
        return a["minutes"] / speed

    ranked = sorted(kb.artifacts, key=lambda a: score(a) / cost(a), reverse=True)
    chosen: list[dict] = []
    for a in ranked:
        trial = chosen + [a]
        if _total(trial, cost) <= minutes:
            chosen = trial

    zone_order = {z: kb.zones[z]["order"] for z in kb.zones}
    chosen.sort(key=lambda a: (zone_order[a["zone"]], a["code"]))

    stops, t, prev_zone = [], 0.0, None
    for a in chosen:
        if prev_zone and a["zone"] != prev_zone:
            t += ZONE_CHANGE_MIN
        stops.append({
            "id": a["id"], "code": a["code"], "name_en": a["name_en"], "name_ar": a["name_ar"],
            "zone": a["zone"], "zone_name_en": kb.zones[a["zone"]]["name_en"],
            "zone_name_ar": kb.zones[a["zone"]]["name_ar"],
            "start_min": round(t), "duration_min": round(cost(a)),
            "matched": sorted(set(a["tags"]) & set(interests)),
            "busy": a["id"] in crowded,
        })
        t += cost(a)
        prev_zone = a["zone"]
    return {"minutes_requested": minutes, "minutes_planned": round(t), "stops": stops}


def _total(items: list[dict], cost) -> float:
    zones = []
    for a in items:
        if a["zone"] not in zones:
            zones.append(a["zone"])
    return sum(cost(a) for a in items) + ZONE_CHANGE_MIN * max(len(zones) - 1, 0)
