"""Artifact knowledge base + a small BM25 retriever (no external dependencies).

The retriever grounds every AI answer in curated facts so the guide never
invents history. In production the JSON seed is swapped for GEM's official
catalogue export; the interface stays the same.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

_AR_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْـ]")
_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)

_STOP = {
    # English
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "is", "was", "it", "for", "with",
    "what", "who", "how", "why", "when", "where", "did", "does", "do", "you", "your", "me",
    "this", "that", "are", "be", "by", "as", "at", "from", "about", "tell",
    # Arabic (MSA + Egyptian)
    "في", "من", "على", "عن", "إلى", "الى", "و", "ده", "دي", "إيه", "ايه", "مين", "ازاي", "إزاي",
    "هو", "هي", "كان", "كانت", "اللي", "يعني", "انت", "إنت", "ليه", "امتى", "فين", "قولي",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = _AR_DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def tokenize(text: str) -> list[str]:
    toks = [t for t in _TOKEN.findall(normalize(text)) if t not in _STOP and len(t) > 1]
    out: list[str] = []
    for t in toks:
        # Strip the Arabic definite article so "القناع" matches "قناع".
        if t.startswith("ال") and len(t) > 4:
            t = t[2:]
        out.append(t)
    return out


class KnowledgeBase:
    def __init__(self, data: dict[str, Any]):
        self.meta = data.get("meta", {})
        self.zones = {z["id"]: z for z in data["zones"]}
        self.tags = data.get("tags", {})
        self.artifacts: list[dict[str, Any]] = data["artifacts"]
        self.by_id = {a["id"]: a for a in self.artifacts}
        self.by_code = {a["code"].upper(): a for a in self.artifacts}
        self._build_index()

    @classmethod
    def from_file(cls, path: Path) -> "KnowledgeBase":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------- retrieval ----------
    def _passages(self) -> list[tuple[str, str]]:
        """(artifact_id, passage) pairs: every fact is its own passage."""
        rows: list[tuple[str, str]] = []
        for a in self.artifacts:
            header = f"{a['name_en']} {a['name_ar']} {a['era_en']} {a['material_en']} {a['material_ar']}"
            rows.append((a["id"], header))
            for fact in a["facts_en"] + a["facts_ar"]:
                rows.append((a["id"], f"{a['name_en']} — {fact}"))
        return rows

    def _build_index(self) -> None:
        self.passages = self._passages()
        self.doc_tokens = [tokenize(p) for _, p in self.passages]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = len(self.doc_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(t) for t in self.doc_tokens]

    def search(self, query: str, k: int = 5, artifact_id: str | None = None,
               k1: float = 1.5, b: float = 0.75) -> list[dict[str, Any]]:
        q = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self.tf):
            aid = self.passages[i][0]
            score = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                score += self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * self.doc_len[i] / self.avgdl))
            if artifact_id and aid == artifact_id:
                score += 0.5  # prefer facts about the artifact being discussed
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [
            {"artifact_id": self.passages[i][0], "text": self.passages[i][1], "score": round(s, 3)}
            for s, i in scored[:k]
        ]

    # ---------- helpers ----------
    def public(self, a: dict[str, Any]) -> dict[str, Any]:
        """Artifact as sent to clients (everything is public in the seed)."""
        zone = self.zones.get(a["zone"], {})
        return {**a, "zone_name_en": zone.get("name_en"), "zone_name_ar": zone.get("name_ar")}

    def fact_sheet(self, a: dict[str, Any]) -> str:
        lines = [
            f"Name: {a['name_en']} / {a['name_ar']}",
            f"Display code: {a['code']}",
            f"Location in museum: {self.zones[a['zone']]['name_en']}",
            f"Era: {a['era_en']}",
            f"Material: {a['material_en']}",
            "Verified facts:",
        ]
        lines += [f"- {f}" for f in a["facts_en"]]
        return "\n".join(lines)


@lru_cache(maxsize=4)
def load_kb(path: str) -> KnowledgeBase:
    return KnowledgeBase.from_file(Path(path))
