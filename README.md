# Hekaya 2 — حكاية ٢

> **Version 2.** Adds a museum guide on top of the original Hekaya:
> - **Interactive museum map**: drag, pinch or scroll to zoom, and tap a numbered pin to fly to a hall, shop, restaurant or service. There's also a link to the real location in Google Maps.
> - **Tickets & visit**: prices (reported from 1 Nov 2026), opening hours, how long a ticket lasts, booking steps on the official site tickets.gem.eg, visit rules and how to get there.
> - **Guide chat**: an AI guide that answers questions about the whole museum (halls, bazaars, food, tickets, hours), not just artifacts. It is grounded in `data/museum.json` and falls back to matching facts offline.
> - New API endpoint: `GET /api/museum`. Chat with `artifact_id: "guide"`.
>
> **New features (Nov 2026):**
> - **Two pieces talk to each other** (Pieces tab): `POST /api/dialogue`. Each piece may only use its own verified facts, and the visitor can join in with a question.
> - **Accessibility mode** (header button): larger text and higher contrast, replies read aloud automatically, voice input with the mic, a "describe yourself" answer built from curated visual descriptions (`visual_cues_ar`/`visual_cues`), and a vibration when a piece opens.
> - **Sounds of ancient Egypt** (from the Sada project): every hall and map place has a live-synthesised soundscape, and the trumpets can be played. Labelled as imagined.
> - **School trips** (Passport tab): the teacher opens a class code, students join with a nickname, and a live leaderboard counts pieces talked to (+10) and quiz answers (+20). `POST /api/class`, `/join`, `/progress`, `GET /api/class/{code}`. Classes expire after 24 h.
> - **Visit souvenir**: a 1080×1920 story card with the visitor's name in hieroglyphs, the pieces they talked to and a line from each. Share or save it.
> - **Museum dashboard** (Passport tab): `GET /api/analytics` shows top pieces, what visitors ask about (topic categories only, never the question text), map places, activity by hour (Cairo time), languages and levels.
> - 27 automated tests.
>
> Visitor info was checked on 4 Nov 2026 from public sources. Prices and hours change, so update `data/museum.json` and confirm on tickets.gem.eg. The map is an illustrative sketch, not to scale.


**Let the Grand Egyptian Museum's artifacts tell their own stories.**
GEM Hackathon 2026 — Fourth Edition · Theme: *How can we make the most of the GEM experience?*

Point your phone at an artifact → Hekaya recognises it → the artifact introduces itself in the first person,
in your language and at your level (kids / visitor / expert), reads it aloud, and answers your questions —
using only curated, verified facts. Add a time-aware personal route and a cartouche "passport" that rewards exploring.

## What's inside

| Path | What it is |
|---|---|
| `data/museum.json` | Museum places for the map, plus tickets, hours, validity, booking and rules (AR/EN) |
| `data/artifacts.json` | Bilingual (AR/EN) knowledge base — 14 GEM highlights, personas, facts, visual cues, display codes |
| `backend/` | FastAPI service: vision recognition, grounded persona chat (streaming), BM25 retrieval, route planner, anonymous analytics |
| `frontend/` | Installable PWA (manifest + service worker), Arabic RTL first, 9 languages, text-to-speech |
| `dist/hekaya-artifact.html` | Same app packaged for the claude.ai live demo |
| `Dockerfile`, `docker-compose.yml` | One-command deployment |

## Run it

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY or GEMINI_API_KEY for AI mode; leave empty for offline mode
uvicorn app.main:app --reload # open http://localhost:8000  (API docs: /docs)
pytest -q                     # 27 tests, run offline
```

Docker: `docker compose up --build`

Rebuild the app after editing the template or the data: `python frontend/build.py`

## How it works

```
Phone (PWA) ──photo──▶ /api/recognize ──▶ Claude vision, constrained to the known artifact list ──▶ artifact id
            ──code───▶ /api/code/GEM-10   (works with no AI and no camera)
            ──question▶ /api/chat/stream ─▶ persona prompt + verified fact sheet + BM25-retrieved related facts
                                             ──▶ Claude ──▶ streamed answer ──▶ browser text-to-speech
            ──────────▶ /api/tour        ──▶ greedy time-budget planner along the museum's flow, avoids busy halls
            ──────────▶ /api/events      ──▶ SQLite (no personal data) ──▶ /api/analytics for the museum team
```

**Three modes, never a dead end:** Claude (full AI) → Hekaya API → offline (curated stories from the embedded knowledge base).

### Guardrails
- Answers are grounded in the fact sheet; the model is told to say "historians aren't sure" instead of inventing.
- Recognition must pick from the known list with confidence ≥ 0.55, otherwise it asks for the display code.
- No photos, names or chat text are stored. Rate limiting per IP. Plain-text output, suitable for reading aloud.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Status and active provider |
| GET | `/api/artifacts?zone=&tag=` | Catalogue |
| GET | `/api/code/{code}` | Lookup by display code (`GEM-10`, `gem10`, `10`) |
| GET | `/api/search?q=` | Arabic/English retrieval |
| POST | `/api/recognize` | Multipart `image` → matched artifact |
| POST | `/api/chat` · `/api/chat/stream` | Persona conversation (JSON / SSE) |
| POST | `/api/tour` | Personal route for minutes + interests |
| POST | `/api/events` · GET `/api/analytics` | Anonymous usage insights |

## Data note
`data/artifacts.json` is a demo seed written for the prototype. Before any real deployment, every entry
must be verified or replaced with GEM's official catalogue and reviewed by curators.

Team: Mohamed El-Saeed Fekry (Moody) — Delta University for Science and Technology, Faculty of AI.
