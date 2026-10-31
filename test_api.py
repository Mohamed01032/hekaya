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
