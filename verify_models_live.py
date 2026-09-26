"""One-shot live verification: are the three routed model IDs REAL and
usable on this project's Vertex project, and does the SOLVE tier return
properly-formed Assamese + LaTeX? Costs a few cents max. Never mutates
anything."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
from google.genai import types
from solver import get_client, _strip_json_fences

PROMPT_SOLVE = (
    'Reply ONLY this JSON, no fences: {"assamese_sentence": "<one short '
    'Assamese sentence stating that the area of a square of side 5 cm is '
    '25 sq cm>", "latex": "\\\\frac{25}{1}"} — write natural textbook-style '
    'Assamese.'
)

def check(name, model_id, prompt, images=None):
    try:
        client = get_client()
        parts = list(images or []) + [prompt]
        resp = client.models.generate_content(
            model=model_id,
            contents=parts,
            config=types.GenerateContentConfig(
                # 512 minimum: gemini-2.5-flash-lite rejects anything lower
                thinking_config=types.ThinkingConfig(thinking_budget=512),
                response_mime_type="application/json"),
        )
        raw = _strip_json_fences(resp.text or "")
        parsed = json.loads(raw)
        sample = json.dumps(parsed, ensure_ascii=False)[:220]
        print(f"[OK]   {name:<8} {model_id:<24} -> {sample}")
        return True
    except Exception as e:
        msg = str(e).replace("\n", " ")[:300]
        print(f"[FAIL] {name:<8} {model_id:<24} -> {msg}")
        return False


if __name__ == "__main__":
    ok_all = True
    ok_all &= check("SOLVE", config.GEMINI_SOLVE_MODEL_ID, PROMPT_SOLVE)
    ok_all &= check("VISION", config.GEMINI_VISION_MODEL_ID,
                    'Reply ONLY this JSON: {"ok": true}')
    ok_all &= check("SCAN", config.GEMINI_SCAN_MODEL_ID,
                    'Reply ONLY this JSON: {"ok": true}')
    print("ALL MODELS LIVE" if ok_all else "ONE OR MORE MODELS FAILED")
    sys.exit(0 if ok_all else 1)
