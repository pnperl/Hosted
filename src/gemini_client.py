from __future__ import annotations

import os
from datetime import datetime, timezone

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def _default_prompt() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "Prepare a concise India market brief for Telegram (max 12 bullets). "
        "Include: overall Indian market sentiment, major national business headline summary, "
        "Gujarat/Jamnagar/Rajkot highlight, Nifty50 outlook, and crude/gold/silver India view. "
        f"Use the most recent context as of {now} and keep language simple."
    )


def _extract_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            chunks: list[str] = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    chunks.append(part_text.strip())
            if chunks:
                return "\n".join(chunks)

    return ""


def get_gemini_prediction(prompt: str | None = None) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing")

    prompt = prompt or _default_prompt()

    # Preferred: new Google GenAI SDK (google-genai)
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = _extract_text(response)
        if text:
            return text
    except Exception:
        pass

    # Backward compatibility: legacy google-generativeai SDK
    try:
        import google.generativeai as gen  # type: ignore

        gen.configure(api_key=GEMINI_API_KEY)

        if hasattr(gen, "GenerativeModel"):
            model = gen.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            text = _extract_text(response)
            if text:
                return text

        # Some legacy installs only expose module-level generation helpers.
        if hasattr(gen, "generate_text"):
            response = gen.generate_text(model=GEMINI_MODEL, prompt=prompt)
            text = _extract_text(response)
            if text:
                return text
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    raise RuntimeError("Gemini request returned no text")
