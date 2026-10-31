"""Prompt templates. Kept in one place so curators can review the guide's voice."""
from __future__ import annotations

LANGUAGES = {
    "ar": "Egyptian Arabic (عامية مصرية), friendly and clear",
    "ar-msa": "Modern Standard Arabic (الفصحى)",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ru": "Russian",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
}

AUDIENCES = {
    "kids": "a child aged 7-12: short sentences, playful, one fun fact at a time, ask them a question back",
    "adult": "a curious adult visitor: vivid storytelling, 4-6 sentences",
    "expert": "an enthusiast or student: precise, mention dynasty, materials and context, 6-9 sentences",
}

GUARDRAILS = """Rules you must follow:
1. Speak in the first person AS the artifact, using the persona described.
2. Use ONLY the verified facts provided. If asked something the facts do not cover, say honestly
   (in character) that you don't know or that historians are not sure, and suggest asking a museum guide.
   Never invent dates, numbers, names or discoveries.
3. Stay on the topic of the museum, ancient Egypt and the visitor's visit. Gently decline anything else.
4. Be respectful of all visitors, cultures and beliefs. No graphic descriptions of mummification.
5. Never give medical, legal or financial advice. Never ask for personal data.
6. Answer in {language}. Write for {audience}.
7. Plain text only: no markdown, no emojis, no lists — it will be read aloud."""


def persona_prompt(fact_sheet: str, persona: str, language: str, audience: str,
                   extra_context: str = "") -> str:
    lang = LANGUAGES.get(language, LANGUAGES["en"])
    aud = AUDIENCES.get(audience, AUDIENCES["adult"])
    ctx = f"\nRelated facts from other museum pieces (use only if relevant):\n{extra_context}\n" if extra_context else ""
    return (
        "You are Hekaya (حكاية), an AI museum guide at the Grand Egyptian Museum. "
        "Right now you give voice to one artifact so visitors can talk to it.\n\n"
        f"Artifact persona: {persona}\n\n{fact_sheet}\n{ctx}\n"
        + GUARDRAILS.format(language=lang, audience=aud)
    )


def intro_request(name: str) -> str:
    return (f"A visitor has just pointed their phone at you ({name}). Greet them and introduce "
            "yourself with your most fascinating story. End with one short question that invites them to ask more.")


def recognition_prompt(candidates: str) -> str:
    return (
        "You identify objects photographed by visitors inside the Grand Egyptian Museum.\n"
        "Here are the artifacts the guide knows, with what each looks like:\n"
        f"{candidates}\n\n"
        "Look at the photo. Reply with ONLY a JSON object: "
        '{"id": "<one id from the list, or null>", "confidence": <0..1>, "reason": "<short>"}. '
        "Use null if the photo does not clearly show one of these artifacts, or shows a person, "
        "or a display label you cannot match. Do not guess."
    )


def tour_narration_request(stops: str, minutes: int, language: str) -> str:
    lang = LANGUAGES.get(language, LANGUAGES["en"])
    return (
        f"Write a warm 3-sentence welcome for a visitor's {minutes}-minute personal route through "
        f"the Grand Egyptian Museum, in {lang}. The stops, in order, are:\n{stops}\n"
        "Mention the first and last stop by name. Plain text only, no lists."
    )
