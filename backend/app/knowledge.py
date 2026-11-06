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
        self.museum: dict[str, Any] = data.get("museum") or {}
        self.places = {p["id"]: p for p in self.museum.get("places", [])}
        if self.museum:
            self.by_id["guide"] = GUIDE
        self._build_index()

    @classmethod
    def from_file(cls, path: Path) -> "KnowledgeBase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        museum = Path(path).with_name("museum.json")
        if museum.exists() and "museum" not in data:
            data["museum"] = json.loads(museum.read_text(encoding="utf-8"))
        return cls(data)

    # ---------- the museum itself (halls, shops, tickets) ----------
    def museum_facts(self) -> list[tuple[str, str, str]]:
        """(source id, language, text) for every place and visitor-info fact."""
        rows: list[tuple[str, str, str]] = []
        for p in self.museum.get("places", []):
            for lang in ("en", "ar"):
                for fact in p[f"facts_{lang}"]:
                    rows.append((f"place:{p['id']}", lang, f"{p[f'name_{lang}']}: {fact}"))
        v = self.museum.get("visit") or {}
        if v:
            for lang in ("en", "ar"):
                cur = "EGP" if lang == "en" else "جنيه"
                prices = "; ".join(f"{t[lang]} {t['egp']} {cur}" if t["egp"] else f"{t[lang]} {'free' if lang == 'en' else 'ببلاش'}"
                                   for t in v.get("tickets", []))
                head = "Ticket prices" if lang == "en" else "أسعار التذاكر"
                rows.append(("visit:tickets", lang, f"{head}: {prices}. {v.get(f'tickets_note_{lang}', '')}"))
                hours = "; ".join(f"{h[lang]}: {'complex' if lang == 'en' else 'المجمع'} {h['complex']}, "
                                  f"{'galleries' if lang == 'en' else 'القاعات'} {h['galleries']}" for h in v.get("hours", []))
                head = "Opening hours" if lang == "en" else "مواعيد الفتح"
                rows.append(("visit:hours", lang, f"{head}: {hours}. {v.get(f'hours_note_{lang}', '')}"))
                for key in ("validity", "booking", "rules"):
                    for line in v.get(f"{key}_{lang}", []):
                        rows.append((f"visit:{key}", lang, line))
                if v.get(f"getting_{lang}"):
                    rows.append(("visit:getting", lang, v[f"getting_{lang}"]))
        return rows

    def guide_sheet(self) -> str:
        m = self.museum
        lines = [f"Information checked on {m.get('checked')}. Official booking site: {m.get('official_booking')}.",
                 m.get("note_en", ""), "Verified facts about the museum:"]
        lines += [f"- [{src}] {text}" for src, lang, text in self.museum_facts() if lang == "en"]
        return "\n".join(lines)

    def guide_answer(self, query: str | None, lang: str, k: int = 3) -> list[str]:
        """Best-matching museum facts in one language (used when no AI model is available)."""
        facts = [(src, text) for src, l, text in self.museum_facts() if l == lang]
        if not query:
            return []
        q = set(tokenize(query))

        def score(src: str, text: str) -> int:
            topic = src.split(":", 1)[1]
            alias = set(tokenize(" ".join(TOPIC_WORDS.get(topic, ()))))
            bonus = 0
            if src.startswith("place:") and src[6:] in self.places:
                if normalize(self.places[src[6:]][f"name_{lang}"]) in normalize(query):
                    bonus = 10  # the visitor named this place
            return len(q & set(tokenize(text))) + 3 * len(q & alias) + bonus

        scored = sorted(((score(src, text), -i) for i, (src, text) in enumerate(facts)), reverse=True)
        if not scored or scored[0][0] == 0:
            return []
        top = facts[-scored[0][1]][0]
        if top.startswith("visit:"):  # practical topics read best as the whole short list, in order
            return [text for src, text in facts if src == top][:4]
        return [facts[-i][1] for sc, i in scored[:k] if sc > 0]

    # ---------- retrieval ----------
    def _passages(self) -> list[tuple[str, str]]:
        """(artifact_id, passage) pairs: every fact is its own passage."""
        rows: list[tuple[str, str]] = []
        for a in self.artifacts:
            header = f"{a['name_en']} {a['name_ar']} {a['era_en']} {a['material_en']} {a['material_ar']}"
            rows.append((a["id"], header))
            for fact in a["facts_en"] + a["facts_ar"]:
                rows.append((a["id"], f"{a['name_en']} — {fact}"))
        rows += [(src, text) for src, _, text in self.museum_facts()]
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
        if a["id"] == "guide":
            return self.guide_sheet()
        lines = [
            f"Name: {a['name_en']} / {a['name_ar']}",
            f"Display code: {a['code']}",
            f"Location in museum: {self.zones[a['zone']]['name_en']}",
            f"Era: {a['era_en']}",
            f"Material: {a['material_en']}",
            f"What it looks like (use when asked to describe yourself, e.g. for blind visitors): {a.get('visual_cues', '')}",
            "Verified facts:",
        ]
        lines += [f"- {f}" for f in a["facts_en"]]
        return "\n".join(lines)


# Everyday words visitors use for each topic (Egyptian Arabic + English), so offline answers still match.
TOPIC_WORDS: dict[str, tuple[str, ...]] = {
    "tickets": ("سعر", "اسعار", "كام", "مصري", "للمصري", "للمصريين", "اجنبي", "للاجانب", "طالب", "طلبه", "بكام", "تمن", "فلوس", "جنيه", "price", "prices", "cost", "fee", "much", "egp"),
    "hours": ("مواعيد", "ميعاد", "بيفتح", "بيقفل", "يفتح", "يقفل", "امتي", "الساعه", "open", "opening", "close", "closing", "hours", "time"),
    "validity": ("مده", "مدتها", "قد", "صلاحيه", "تنفع", "ساعات", "اقعد", "خروج", "valid", "validity", "stay", "long", "reentry", "refund"),
    "booking": ("احجز", "حجز", "اشتري", "اونلاين", "موقع", "book", "booking", "buy", "online", "website"),
    "rules": ("تصوير", "فلاش", "ممنوع", "مسموح", "صور", "photo", "photos", "camera", "flash", "allowed", "rules", "tripod"),
    "getting": ("اروح", "اوصل", "مواصلات", "عنوان", "مكان", "تاكسي", "اوبر", "get", "reach", "address", "taxi", "metro", "transport"),
    "shops": ("محلات", "بازار", "بازارات", "هدايا", "تذكارات", "شوبينج", "shop", "shops", "shopping", "souvenir", "souvenirs", "gift", "bazaar"),
    "food": ("اكل", "مطعم", "مطاعم", "كافيه", "قهوه", "اشرب", "جعان", "food", "eat", "restaurant", "cafe", "coffee", "drink"),
    "parking": ("جراج", "اركن", "ركنه", "عربيه", "parking", "park", "car"),
    "gate": ("امن", "تفتيش", "شباك", "مدخل", "security", "entrance", "gate"),
    "children": ("اطفال", "طفل", "عيال", "ولادي", "children", "kids", "child"),
    "services": ("اسانسير", "كرسي", "متحرك", "حمام", "استعلامات", "lift", "elevator", "wheelchair", "toilet", "help", "information"),
    "tutankhamun": ("توت", "tut"),
    "khufu-boats": ("مركب", "مراكب", "boat", "boats"),
    "grand-staircase": ("سلم", "الدرج", "stairs", "staircase"),
    "galleries": ("قاعات", "القاعات", "galleries", "halls", "rooms"),
    "gardens": ("حدايق", "جنينه", "garden", "gardens"),
}

PRACTICAL = ("tickets", "hours", "validity", "booking", "rules", "getting", "shops", "food",
             "parking", "gate", "children", "services")
DESCRIBE_WORDS = {"صف", "اوصف", "وصف", "شكلك", "شكله", "شكل", "لونك", "describe", "look", "looks", "appearance", "colour", "color"}


def topic_of(message: str | None) -> str | None:
    """Coarse category of a visitor question, for anonymous analytics (never the text itself)."""
    if not message:
        return None
    q = set(tokenize(message))
    if q & set(tokenize(" ".join(DESCRIBE_WORDS))):
        return "describe"
    best, score = None, 0
    for topic in PRACTICAL:
        n = len(q & set(tokenize(" ".join(TOPIC_WORDS.get(topic, ())))))
        if n > score:
            best, score = topic, n
    return best or "history"


GUIDE: dict[str, Any] = {
    "id": "guide", "code": "GUIDE", "zone": "plaza", "kind": "guide",
    "name_en": "Hekaya, your museum guide", "name_ar": "حكاية، مرشدتك في المتحف",
    "persona_en": "A warm, practical guide who knows every hall, shop, ticket rule and shortcut of the museum.",
    "persona_ar": "مرشدة ودودة وعملية، عارفة كل قاعة ومحل وقواعد التذاكر وأقصر طريق في المتحف.",
}


@lru_cache(maxsize=4)
def load_kb(path: str) -> KnowledgeBase:
    return KnowledgeBase.from_file(Path(path))
