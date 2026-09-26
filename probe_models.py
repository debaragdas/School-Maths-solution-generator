"""Probe which Gemini model IDs are actually available on this project."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config
from google.genai import types
from solver import get_client

CANDIDATES = [
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemini-3-flash", "gemini-3-flash-preview", "gemini-3-pro",
    "gemini-3.1-flash-lite", "gemini-3.1-pro", "gemini-2.0-flash",
]

client = get_client()
for m in CANDIDATES:
    try:
        client.models.generate_content(
            model=m,
            contents=['Reply ONLY JSON: {"ok": true}'],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=512),
                response_mime_type="application/json"),
        )
        print(f"[AVAILABLE] {m}")
    except Exception as e:
        code = getattr(e, "code", None) or ""
        print(f"[no]        {m}  ({str(e)[:80]})")
