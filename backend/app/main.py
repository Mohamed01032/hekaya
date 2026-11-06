"""Hekaya API — the AI museum guide backend for the Grand Egyptian Museum.

Run:  uvicorn app.main:app --reload   (from the backend/ folder)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import prompts
from .analytics import Analytics
from .config import ROOT, get_settings
from .classes import Classes
from .knowledge import DESCRIBE_WORDS, load_kb, tokenize, topic_of
from .llm import build_provider, parse_json
from .tour import plan_tour

settings = get_settings()
kb = load_kb(str(settings.data_path))
llm = build_provider(settings)
analytics = Analytics(settings.db_path)
classes = Classes(settings.db_path)

app = FastAPI(
    title="Hekaya — حكاية",
    description="AI guide that lets Grand Egyptian Museum artifacts tell their own stories.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"],
)

Language = Literal["ar", "ar-msa", "en", "fr", "de", "es", "it", "ru", "zh", "ja"]
Audience = Literal["kids", "adult", "expert"]


# ---------------------------------------------------------------- intro cache
# Introductions are identical for every visitor of the same piece, language and level,
# so they are generated once and reused (cuts AI cost on the most common call).
_intro_cache: dict[tuple[str, str, str], str] = {}

# ---------------------------------------------------------------- rate limiting
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now, window = time.time(), _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_min:
        raise HTTPException(429, "Too many requests — please wait a moment.")
    window.append(now)


# ---------------------------------------------------------------- schemas
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatIn(BaseModel):
    artifact_id: str
    message: str | None = Field(default=None, max_length=1000,
                                description="Visitor question; omit to get the artifact's introduction")
    history: list[Turn] = Field(default_factory=list, max_length=20)
    language: Language = "ar"
    audience: Audience = "adult"


class ChatOut(BaseModel):
    artifact_id: str
    reply: str
    sources: list[str]
    provider: str


class TourIn(BaseModel):
    minutes: int = Field(ge=15, le=480)
    interests: list[str] = Field(default_factory=list)
    audience: Audience = "adult"
    language: Language = "ar"
    crowded: list[str] = Field(default_factory=list)
    narrate: bool = False


class EventIn(BaseModel):
    kind: str = Field(max_length=24)
    artifact_id: str | None = Field(default=None, max_length=64)
    language: str | None = Field(default=None, max_length=8)
    audience: str | None = Field(default=None, max_length=8)


class Line(BaseModel):
    who: Literal["a", "b", "visitor"]
    text: str = Field(min_length=1, max_length=1000)


class DialogueIn(BaseModel):
    a: str
    b: str
    topic: str | None = Field(default=None, max_length=80)
    question: str | None = Field(default=None, max_length=300)
    history: list[Line] = Field(default_factory=list, max_length=30)
    language: Language = "ar"
    audience: Audience = "adult"


class DialogueOut(BaseModel):
    lines: list[dict]
    provider: str


class ClassIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class JoinIn(BaseModel):
    nick: str = Field(min_length=1, max_length=24)


class ProgressIn(BaseModel):
    sid: str = Field(min_length=4, max_length=40)
    stamps: int = Field(ge=0, le=200)
    quiz: int = Field(ge=0, le=200)


# ---------------------------------------------------------------- helpers
def _artifact(artifact_id: str) -> dict:
    a = kb.by_id.get(artifact_id)
    if not a:
        raise HTTPException(404, f"Unknown artifact '{artifact_id}'")
    return a


def _offline_reply(body: "ChatIn") -> str:
    """Curated answer in the visitor's language when no AI model is configured."""
    a = _artifact(body.artifact_id)
    arabic = body.language.startswith("ar")
    if a["id"] == "guide":
        lang = "ar" if arabic else "en"
        picked = kb.guide_answer(body.message, lang)
        if not body.message:
            return ("أهلاً بيك! أنا حكاية، مرشدتك في المتحف. اسألني عن القاعات والمحلات والأكل والتذاكر والمواعيد."
                    if arabic else "Welcome! I am Hekaya, your museum guide. Ask me about halls, shops, food, tickets and hours.")
        if not picked:
            return ("المعلومة دي مش عندي. اسأل مكتب الاستعلامات في المتحف، أو شوف الموقع الرسمي "
                    if arabic else "I don't have that information. Please ask the museum information desk or check ") + \
                kb.museum.get("official_booking", "")
        return " ".join(picked)
    facts = a["facts_ar"] if arabic else a["facts_en"]
    if body.message and set(tokenize(body.message)) & set(tokenize(" ".join(DESCRIBE_WORDS))):
        return (a.get("visual_cues_ar") if arabic else a.get("visual_cues")) or ""
    if body.message:
        hits = [h["text"] for h in kb.search(body.message, k=8, artifact_id=a["id"])
                if h["artifact_id"] == a["id"]]
        # keep only hits in the visitor's language, mapped back to the plain fact text
        picked = [f for f in facts if any(f in h for h in hits)][:2] or facts[:2]
        lead = ""
    else:
        picked = facts[:2]
        lead = f"أهلاً بيك! أنا {a['name_ar']}. " if arabic else f"Hello! I am {a['name_en']}. "
    note = (" (بحكيلك من المعلومات الموثقة بس لأن الذكاء الاصطناعي مش متاح دلوقتي.)" if arabic
            else " (Telling you from verified facts only, because the AI is unavailable right now.)")
    return lead + " ".join(picked) + note


def _ai_error(exc: Exception) -> str:
    """One-line reason an AI call failed (bad key, unknown model, quota), shown to the user."""
    msg = " ".join(str(exc).split())[:220]
    return f"AI service error ({llm.name}): {type(exc).__name__}: {msg}"


def _build_chat(body: ChatIn) -> tuple[str, list[dict], list[str]]:
    a = _artifact(body.artifact_id)
    if a["id"] == "guide":
        system = prompts.guide_prompt(kb.guide_sheet(), body.language, body.audience,
                                      kb.museum.get("checked", ""), kb.museum.get("official_booking", ""))
        turns = [t.model_dump() for t in body.history][-12:]
        turns.append({"role": "user", "content": body.message or prompts.guide_intro_request()})
        if turns[0]["role"] != "user":
            turns = turns[1:]
        return system, turns, ["museum"]
    # Arabic personas are written in Egyptian dialect, so Fusha (ar-msa) uses the English persona
    persona = a["persona_ar"] if body.language == "ar" else a["persona_en"]
    related, sources = "", [a["id"]]
    if body.message:
        hits = [h for h in kb.search(body.message, k=4, artifact_id=a["id"]) if h["artifact_id"] != a["id"]]
        related = "\n".join(f"- {h['text']}" for h in hits[:3])
        sources += sorted({h["artifact_id"] for h in hits[:3]})
    system = prompts.persona_prompt(kb.fact_sheet(a), persona, body.language, body.audience, related)
    turns = [t.model_dump() for t in body.history][-12:]
    turns.append({"role": "user", "content": body.message or prompts.intro_request(a["name_en"])})
    if turns[0]["role"] != "user":
        turns = turns[1:]
    return system, turns, sources


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"ok": True, "provider": llm.name, "artifacts": len(kb.artifacts), "version": app.version}


@app.get("/api/museum")
def museum_info():
    """Map places, tickets, hours and booking steps (illustrative map, check prices on the official site)."""
    return kb.museum


@app.get("/api/artifacts")
def list_artifacts(zone: str | None = None, tag: str | None = None):
    items = kb.artifacts
    if zone:
        items = [a for a in items if a["zone"] == zone]
    if tag:
        items = [a for a in items if tag in a["tags"]]
    return {"zones": list(kb.zones.values()), "tags": kb.tags, "artifacts": [kb.public(a) for a in items]}


@app.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: str):
    return kb.public(_artifact(artifact_id))


@app.get("/api/code/{code}")
def by_code(code: str):
    """Fallback when the camera can't recognise a piece: visitor types the code on the display label."""
    a = kb.by_code.get(code.strip().upper().replace(" ", ""))
    if not a and code.strip().isdigit():
        a = kb.by_code.get(f"GEM-{int(code):02d}")
    if not a:
        raise HTTPException(404, "No artifact with that display code")
    analytics.log("scan", a["id"])
    return kb.public(a)


@app.get("/api/search")
def search(q: str, k: int = 5):
    return {"query": q, "results": kb.search(q, k=min(k, 20))}


@app.post("/api/recognize", dependencies=[Depends(rate_limit)])
async def recognize(image: UploadFile = File(...), language: str = Form("ar")):
    data = await image.read()
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, "Image too large (max 8 MB)")
    media = image.content_type or "image/jpeg"
    if media not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise HTTPException(415, "Use a JPEG, PNG, WebP or GIF photo")
    candidates = "\n".join(f"- id: {a['id']} | {a['name_en']} | looks like: {a['visual_cues']}"
                           for a in kb.artifacts)
    try:
        raw = llm.complete("", [{"role": "user", "content": prompts.recognition_prompt(candidates)}],
                           images=[(data, media)], fast=False, max_tokens=200)
    except Exception as exc:
        raise HTTPException(502, _ai_error(exc))
    result = parse_json(raw) or {}
    aid, conf = result.get("id"), float(result.get("confidence") or 0)
    if aid in kb.by_id and conf >= 0.55:
        analytics.log("scan", aid, language)
        return {"match": kb.public(kb.by_id[aid]), "confidence": conf, "reason": result.get("reason", "")}
    analytics.log("recognize_fail", None, language)
    return {"match": None, "confidence": conf, "reason": result.get("reason", "not recognised"),
            "hint": "Try a closer photo of the object, or type the code on its display label."}


@app.post("/api/chat", response_model=ChatOut, dependencies=[Depends(rate_limit)])
def chat(body: ChatIn):
    system, turns, sources = _build_chat(body)
    key = (body.artifact_id, body.language, body.audience)
    is_intro = not body.message and not body.history
    if is_intro and key in _intro_cache:
        reply = _intro_cache[key]
    else:
        try:
            reply = _offline_reply(body) if llm.name == "offline" else llm.complete(system, turns, fast=body.audience == "kids")
        except Exception as exc:
            raise HTTPException(502, _ai_error(exc))
        if is_intro and llm.name != "offline":
            _intro_cache[key] = reply
    analytics.log("guide" if body.artifact_id == "guide" else "chat", body.artifact_id, body.language,
                  body.audience, topic_of(body.message))
    return ChatOut(artifact_id=body.artifact_id, reply=reply, sources=sources, provider=llm.name)


@app.post("/api/chat/stream", dependencies=[Depends(rate_limit)])
def chat_stream(body: ChatIn):
    """Server-sent events: `data: <text chunk>` lines, then `event: done`."""
    system, turns, _ = _build_chat(body)
    analytics.log("guide" if body.artifact_id == "guide" else "chat", body.artifact_id, body.language,
                  body.audience, topic_of(body.message))

    key = (body.artifact_id, body.language, body.audience)
    is_intro = not body.message and not body.history

    def gen():
        try:
            if is_intro and key in _intro_cache:
                yield "data: " + _intro_cache[key].replace("\n", "\\n") + "\n\n"
                yield "event: done\ndata: \n\n"
                return
            parts: list[str] = []
            chunks = (w + " " for w in _offline_reply(body).split(" ")) if llm.name == "offline" \
                else llm.stream(system, turns, fast=body.audience == "kids")
            for chunk in chunks:
                parts.append(chunk)
                yield "data: " + chunk.replace("\n", "\\n") + "\n\n"
            if is_intro and llm.name != "offline":
                _intro_cache[key] = "".join(parts).strip()
            yield "event: done\ndata: \n\n"
        except Exception as exc:  # surface errors to the client instead of hanging
            yield f"event: error\ndata: {_ai_error(exc)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _offline_dialogue(body: DialogueIn, a: dict, b: dict) -> list[dict]:
    arabic = body.language.startswith("ar")
    fa, fb = (a["facts_ar"], b["facts_ar"]) if arabic else (a["facts_en"], b["facts_en"])
    said = {"a": sum(1 for x in body.history if x.who == "a"), "b": sum(1 for x in body.history if x.who == "b")}
    if body.question:
        def best(art, facts):
            hits = [h["text"] for h in kb.search(body.question, k=8, artifact_id=art["id"]) if h["artifact_id"] == art["id"]]
            return next((f for f in facts if any(f in h for h in hits)), facts[said["a"] % len(facts)])
        return [{"who": "a", "text": best(a, fa)}, {"who": "b", "text": best(b, fb)}]
    na, nb = a["name_ar" if arabic else "name_en"], b["name_ar" if arabic else "name_en"]
    i, j = said["a"], said["b"]
    if not body.history:
        opener_a = f"أهلاً يا {nb}! أنا {na}. " if arabic else f"Hello, {nb}! I am {na}. "
        opener_b = f"وأنا {nb}، يا أهلاً بيك. " if arabic else f"And I am {nb}, welcome. "
    else:
        opener_a = "وكمان، " if arabic else "Also, "
        opener_b = "وأنا كمان عندي حكاية: " if arabic else "I have a story too: "
    ask_b = (f" وإنت يا {nb}؟" if arabic else f" And you, {nb}?")
    lines = [{"who": "a", "text": opener_a + fa[i % len(fa)] + ask_b},
             {"who": "b", "text": opener_b + fb[j % len(fb)]},
             {"who": "a", "text": fa[(i + 1) % len(fa)]},
             {"who": "b", "text": fb[(j + 1) % len(fb)]}]
    return lines


def _clean_dialogue(data: dict | None) -> list[dict] | None:
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        return None
    out = [{"who": x["who"], "text": " ".join(str(x.get("text", "")).split())[:500]}
           for x in data["lines"][:6] if isinstance(x, dict) and x.get("who") in ("a", "b") and x.get("text")]
    return out or None


@app.post("/api/dialogue", response_model=DialogueOut, dependencies=[Depends(rate_limit)])
def dialogue(body: DialogueIn):
    """Two artifacts talk to each other; the visitor can join with a question."""
    a, b = _artifact(body.a), _artifact(body.b)
    if a["id"] == b["id"] or "guide" in (a["id"], b["id"]):
        raise HTTPException(400, "Pick two different artifacts.")
    analytics.log("dialogue", a["id"], body.language, body.audience, topic_of(body.question))
    analytics.log("dialogue", b["id"], body.language, body.audience)
    lines = None
    if llm.name != "offline":
        arabic = body.language.startswith("ar")
        system = prompts.dialogue_prompt(kb.fact_sheet(a), kb.fact_sheet(b),
                                         a["persona_ar" if body.language == "ar" else "persona_en"], b["persona_ar" if body.language == "ar" else "persona_en"],
                                         body.language, body.audience, body.topic)
        names = {"a": a["name_en"], "b": b["name_en"], "visitor": "Visitor"}
        transcript = "\n".join(f"{names[x.who]}: {x.text}" for x in body.history[-12:])
        ask = (f"Conversation so far:\n{transcript}\n\n" if transcript else "Start the conversation.\n") + \
              (f"The visitor now asks: {body.question}\n" if body.question else "Continue naturally.\n")
        try:
            lines = _clean_dialogue(parse_json(llm.complete(system, [{"role": "user", "content": ask}],
                                                            fast=True, max_tokens=900)))
        except Exception:
            lines = None
    used = llm.name if lines else "offline"
    return DialogueOut(lines=lines or _offline_dialogue(body, a, b), provider=used)


# ---------------------------------------------------------------- school trips
def _code(code: str) -> str:
    c = code.strip().upper()
    if not (4 <= len(c) <= 8 and c.isalnum()):
        raise HTTPException(400, "Bad class code.")
    return c


@app.post("/api/class", dependencies=[Depends(rate_limit)])
def class_create(body: ClassIn):
    return {"code": classes.create(" ".join(body.name.split()))}


@app.post("/api/class/{code}/join", dependencies=[Depends(rate_limit)])
def class_join(code: str, body: JoinIn):
    sid = classes.join(_code(code), " ".join(body.nick.split()))
    if not sid:
        raise HTTPException(404, "Class not found, expired or full.")
    analytics.log("class_join")
    return {"sid": sid}


@app.post("/api/class/{code}/progress", status_code=204, dependencies=[Depends(rate_limit)])
def class_progress(code: str, body: ProgressIn):
    if not classes.progress(_code(code), body.sid, body.stamps, body.quiz):
        raise HTTPException(404, "Class or student not found.")


@app.get("/api/class/{code}")
def class_board(code: str):
    board = classes.board(_code(code))
    if not board:
        raise HTTPException(404, "Class not found or expired.")
    return board


@app.post("/api/tour")
def tour(body: TourIn):
    plan = plan_tour(kb, body.minutes, body.interests, body.audience, body.crowded)
    if body.narrate and plan["stops"]:
        names = "\n".join(f"{i+1}. {s['name_en']} ({s['zone_name_en']})" for i, s in enumerate(plan["stops"]))
        try:
            plan["welcome"] = llm.complete(
                "", [{"role": "user", "content": prompts.tour_narration_request(names, body.minutes, body.language)}],
                fast=True, max_tokens=250)
        except Exception:
            plan["welcome"] = None
    analytics.log("tour", None, body.language, body.audience)
    return plan


@app.post("/api/events", status_code=204)
def event(body: EventIn):
    analytics.log(body.kind, body.artifact_id, body.language, body.audience)


@app.get("/api/analytics")
def analytics_summary(hours: float = 168):
    return analytics.summary(hours)


# ---------------------------------------------------------------- serve the PWA
FRONTEND = ROOT / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(FRONTEND) / "index.html")

    @app.get("/{name}", include_in_schema=False)
    def root_file(name: str):
        f = Path(FRONTEND) / name
        if name in {"manifest.webmanifest", "sw.js", "icon.svg", "artifacts.json"} and f.exists():
            return FileResponse(f)
        if name == "artifacts.json":
            return FileResponse(settings.data_path)
        raise HTTPException(404)
