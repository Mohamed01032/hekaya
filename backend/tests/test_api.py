"""API tests — run in offline mode (no API key needed): `pytest -q` from backend/."""
import io
import os
import tempfile

os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["HEKAYA_DB"] = os.path.join(tempfile.mkdtemp(), "test.sqlite3")
os.environ["HEKAYA_RATE_LIMIT"] = "1000"

from fastapi.testclient import TestClient  # noqa: E402

from app.knowledge import tokenize  # noqa: E402
from app.main import app, kb  # noqa: E402
import app.main as m  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["provider"] == "offline"
    assert r.json()["artifacts"] == len(kb.artifacts) >= 10


def test_every_artifact_is_complete():
    required = {"id", "code", "zone", "minutes", "tags", "name_en", "name_ar", "facts_en", "facts_ar",
                "persona_en", "persona_ar", "visual_cues", "kid_hook_en", "kid_hook_ar"}
    codes = set()
    for a in kb.artifacts:
        assert required <= a.keys(), a["id"]
        assert a["zone"] in kb.zones
        assert len(a["facts_en"]) == len(a["facts_ar"]) >= 3, a["id"]
        assert all(t in kb.tags for t in a["tags"]), a["id"]
        assert a["code"] not in codes
        codes.add(a["code"])


def test_list_and_filter():
    assert len(client.get("/api/artifacts").json()["artifacts"]) == len(kb.artifacts)
    tut = client.get("/api/artifacts", params={"zone": "tutankhamun"}).json()["artifacts"]
    assert tut and all(a["zone"] == "tutankhamun" for a in tut)


def test_code_lookup_variants():
    assert client.get("/api/code/GEM-10").json()["id"] == "tut-mask"
    assert client.get("/api/code/gem-10").json()["id"] == "tut-mask"
    assert client.get("/api/code/10").json()["id"] == "tut-mask"
    assert client.get("/api/code/GEM-99").status_code == 404


def test_arabic_normalisation():
    assert tokenize("القناع الذهبي") == tokenize("القناع الذهبى")


def test_search_arabic_and_english():
    assert client.get("/api/search", params={"q": "meteorite iron blade"}).json()["results"][0]["artifact_id"] == "tut-dagger"
    assert client.get("/api/search", params={"q": "مركب خوفو الأرز"}).json()["results"][0]["artifact_id"] == "khufu-boat"


def test_chat_intro_and_question_offline():
    r = client.post("/api/chat", json={"artifact_id": "tut-mask", "language": "en"})
    assert r.status_code == 200 and "Golden Mask" in r.json()["reply"]
    r = client.post("/api/chat", json={"artifact_id": "tut-mask", "language": "ar",
                                       "message": "مين اكتشف المقبرة؟"})
    assert r.status_code == 200 and r.json()["reply"]
    assert client.post("/api/chat", json={"artifact_id": "nope"}).status_code == 404


def test_chat_stream():
    with client.stream("POST", "/api/chat/stream", json={"artifact_id": "khufu-boat", "language": "en"}) as r:
        body = "".join(r.iter_text())
    assert "data:" in body and "event: done" in body


def test_recognize_offline_falls_back():
    png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f"
           b"\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82")
    r = client.post("/api/recognize", files={"image": ("x.png", io.BytesIO(png), "image/png")})
    assert r.status_code == 200 and r.json()["match"] is None and "code" in r.json()["hint"]
    bad = client.post("/api/recognize", files={"image": ("x.txt", io.BytesIO(b"hi"), "text/plain")})
    assert bad.status_code == 415


def test_tour_respects_time_and_flow():
    for minutes in (30, 60, 120):
        plan = client.post("/api/tour", json={"minutes": minutes, "interests": ["science"]}).json()
        assert plan["minutes_planned"] <= minutes and plan["stops"]
        orders = [kb.zones[s["zone"]]["order"] for s in plan["stops"]]
        assert orders == sorted(orders)
    plan = client.post("/api/tour", json={"minutes": 45, "interests": ["science", "space"]}).json()
    assert "tut-dagger" in [s["id"] for s in plan["stops"]]


def test_tour_avoids_crowds():
    calm = client.post("/api/tour", json={"minutes": 30, "interests": []}).json()
    busy = client.post("/api/tour", json={"minutes": 30, "interests": [], "crowded": ["tut-mask"]}).json()
    assert [s["id"] for s in calm["stops"]] != [s["id"] for s in busy["stops"]] or \
        any(s["busy"] for s in busy["stops"])


def test_analytics_records_without_personal_data():
    client.post("/api/events", json={"kind": "listen", "artifact_id": "tut-mask", "language": "fr"})
    client.post("/api/events", json={"kind": "drop table", "artifact_id": "x"})  # ignored
    s = client.get("/api/analytics").json()
    assert s["events"].get("listen", 0) >= 1 and "drop table" not in s["events"]


def test_offline_answers_in_visitor_language():
    r = client.post("/api/chat", json={"artifact_id": "khufu-boat", "language": "ar",
                                       "message": "مين اكتشف المركب؟"}).json()["reply"]
    assert "كمال الملاخ" in r and "discovered" not in r


def test_intro_cache_reused(monkeypatch):
    import app.main as m
    calls = []

    class Fake:
        name = "fake"
        def complete(self, system, turns, **kw):
            calls.append(1); return "intro text"
        def stream(self, system, turns, **kw):
            yield "intro text"

    monkeypatch.setattr(m, "llm", Fake())
    m._intro_cache.clear()
    for _ in range(3):
        assert client.post("/api/chat", json={"artifact_id": "tut-throne", "language": "fr"}).json()["reply"] == "intro text"
    assert len(calls) == 1
    m._intro_cache.clear()


def test_gemini_provider_builds_requests(monkeypatch):
    from app.llm import GeminiProvider, build_provider
    from app.config import Settings
    sent = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            sent.update(model=model, contents=contents, config=config)
            return type("R", (), {"text": " salam "})()
        def generate_content_stream(self, model, contents, config):
            for t in ("a", None, "b"):
                yield type("C", (), {"text": t})()

    p = GeminiProvider("k", "big", "small")
    p.client = type("Cl", (), {"models": FakeModels()})()
    out = p.complete("SYS", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                             {"role": "user", "content": "what?"}], images=[(b"img", "image/png")], fast=True)
    assert out == "salam" and sent["model"] == "small"
    assert [c.role for c in sent["contents"]] == ["user", "model", "user"]
    assert len(sent["contents"][-1].parts) == 2 and sent["config"].system_instruction == "SYS"
    assert "".join(p.stream("S", [{"role": "user", "content": "x"}])) == "ab"

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert build_provider(Settings()).name == "gemini"
    monkeypatch.setenv("HEKAYA_PROVIDER", "offline")
    assert build_provider(Settings()).name == "offline"


def test_ai_failure_returns_readable_error(monkeypatch):
    import app.main as m

    class Broken:
        name = "gemini"
        def complete(self, *a, **k):
            raise RuntimeError("404 model not found")
        def stream(self, *a, **k):
            raise RuntimeError("404 model not found")
            yield ""

    monkeypatch.setattr(m, "llm", Broken())
    m._intro_cache.clear()
    r = client.post("/api/chat", json={"artifact_id": "tut-mask", "language": "en"})
    assert r.status_code == 502 and "model not found" in r.json()["detail"]
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
    r = client.post("/api/recognize", files={"image": ("x.png", io.BytesIO(png), "image/png")})
    assert r.status_code == 502 and "gemini" in r.json()["detail"]


# ---------------------------------------------------------------- museum guide (v2)
def test_museum_endpoint_has_places_and_tickets():
    m = client.get("/api/museum").json()
    ids = {p["id"] for p in m["places"]}
    assert {"grand-hall", "tutankhamun", "shops", "food", "gate"} <= ids
    assert m["official_booking"].startswith("https://") and m["visit"]["tickets"]


def test_guide_offline_answers_tickets_in_arabic():
    r = client.post("/api/chat", json={"artifact_id": "guide", "message": "سعر التذكرة للمصري كام؟", "language": "ar"}).json()
    assert r["provider"] == "offline" and "220" in r["reply"]


def test_guide_offline_answers_shops_in_english():
    r = client.post("/api/chat", json={"artifact_id": "guide", "message": "where are the shops and souvenirs?", "language": "en"}).json()
    assert "shop" in r["reply"].lower()


def test_guide_unknown_question_is_honest():
    r = client.post("/api/chat", json={"artifact_id": "guide", "message": "xyzzy qwerty", "language": "en"}).json()
    assert "information desk" in r["reply"]


def test_guide_prompt_is_grounded():
    body = m.ChatIn(artifact_id="guide", message="hours?", language="en")
    system, turns, sources = m._build_chat(body)
    assert "Never invent prices" in system and "tickets.gem.eg" in system and sources == ["museum"]


def test_artifact_chat_can_use_museum_facts():
    body = m.ChatIn(artifact_id="tut-mask", message="what time does the museum close on Wednesday hours", language="en")
    system, _, sources = m._build_chat(body)
    assert any(s.startswith("visit:") or s.startswith("place:") for s in sources)


# ---------------------------------------------------------------- v2 features
def test_dialogue_offline_alternates_and_uses_own_facts():
    r = client.post("/api/dialogue", json={"a": "tut-mask", "b": "khufu-boat", "language": "en"}).json()
    whos = [x["who"] for x in r["lines"]]
    assert r["provider"] == "offline" and whos == ["a", "b", "a", "b"]
    mask, boat = kb.by_id["tut-mask"], kb.by_id["khufu-boat"]
    assert any(f in r["lines"][0]["text"] for f in mask["facts_en"])
    assert any(f in r["lines"][1]["text"] for f in boat["facts_en"])


def test_dialogue_question_and_validation():
    r = client.post("/api/dialogue", json={"a": "tut-mask", "b": "tut-throne", "question": "ليه اتعملتوا؟",
                                           "history": [{"who": "a", "text": "أهلاً"}], "language": "ar"}).json()
    assert len(r["lines"]) == 2
    assert client.post("/api/dialogue", json={"a": "tut-mask", "b": "tut-mask"}).status_code == 400
    assert client.post("/api/dialogue", json={"a": "tut-mask", "b": "nope"}).status_code == 404


def test_describe_offline_returns_appearance():
    r = client.post("/api/chat", json={"artifact_id": "tut-mask", "message": "صف شكلك", "language": "ar"}).json()
    assert r["reply"] == kb.by_id["tut-mask"]["visual_cues_ar"]


def test_school_class_flow():
    code = client.post("/api/class", json={"name": "Grade 5"}).json()["code"]
    s1 = client.post(f"/api/class/{code}/join", json={"nick": "Mona"}).json()["sid"]
    s2 = client.post(f"/api/class/{code}/join", json={"nick": "Omar"}).json()["sid"]
    assert client.post(f"/api/class/{code}/progress", json={"sid": s1, "stamps": 2, "quiz": 1}).status_code == 204
    assert client.post(f"/api/class/{code}/progress", json={"sid": s2, "stamps": 5, "quiz": 0}).status_code == 204
    board = client.get(f"/api/class/{code.lower()}").json()
    assert [s["nick"] for s in board["students"]] == ["Omar", "Mona"] and board["students"][0]["score"] == 50
    assert client.post("/api/class/ZZZZZ/join", json={"nick": "x"}).status_code == 404
    assert client.post(f"/api/class/{code}/progress", json={"sid": "fake-sid", "stamps": 1, "quiz": 0}).status_code == 404


def test_analytics_counts_topics_not_text():
    client.post("/api/chat", json={"artifact_id": "guide", "message": "التذكرة بكام؟", "language": "ar"})
    client.post("/api/events", json={"kind": "place", "artifact_id": "shops"})
    a = client.get("/api/analytics").json()
    assert any(t["topic"] == "tickets" for t in a["topics"]) and len(a["hours"]) == 24
    assert any(p["id"] == "shops" for p in a["places"]) and a["total"] > 0
    import sqlite3
    rows = sqlite3.connect(m.settings.db_path).execute("SELECT * FROM events").fetchall()
    assert not any("بكام" in str(r) for r in rows)  # question text is never stored
