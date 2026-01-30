import os
import re
import json
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai

# -------------------------
# Config
# -------------------------

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")

DATA_DIR = Path("data/processed_dialogues")
CACHE_DIR = Path("cache/gemini")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MASK_PATTERN = re.compile(r"\[([A-Z][A-Z0-9_]{1,50})\]")

# -------------------------
# Helpers
# -------------------------

def extract_masks_from_dialogue(dialogue: list[dict]) -> list[str]:
    """Extract unique mask names from all turns"""
    masks = set()
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        text = turn.get("text", "")
        masks.update(MASK_PATTERN.findall(text))
    return sorted(masks)


def call_gemini(dialogue_text: str, masks: list[str]) -> dict:
    prompt = f"""
You are filling masked fields in call center dialogues.

Context:
- Region: South Asia (India, Vietnam, Philippines, etc.)
- Language: English
- Vietnamese names MUST keep diacritical marks (e.g., Hiếu, Thảo, Nguyễn)
- Values must be realistic for call centers
- Use diverse names, companies, roles
- Do NOT invent new fields
- Do NOT rewrite the dialogue
- Do NOT change language
- Return JSON ONLY

Masked fields to fill:
{", ".join(masks)}

Dialogue:
\"\"\"{dialogue_text}\"\"\"

Return JSON only in this format:
{{
  "FIELD_NAME": "value"
}}
"""
    response = model.generate_content(prompt)

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        raise ValueError(f"Gemini returned invalid JSON:\n{response.text}")


def replace_masks_in_turn(text: str, values: dict) -> str:
    for k, v in values.items():
        text = text.replace(f"[{k}]", v)
    return text


# -------------------------
# Main processing
# -------------------------

def process_dialogue(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        dialogue = json.load(f)

    # Ensure dialogue is a list of turns
    if not isinstance(dialogue, list):
        raise ValueError("Dialogue JSON is not a list")

    masks = extract_masks_from_dialogue(dialogue)
    if not masks:
        return  # nothing to fill

    # Build full dialogue text for Gemini
    full_text = " ".join(
        turn.get("text", "")
        for turn in dialogue
        if isinstance(turn, dict)
    )

    cache_path = CACHE_DIR / f"{path.stem}.json"

    if cache_path.exists():
        values = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        values = call_gemini(full_text, masks)
        cache_path.write_text(
            json.dumps(values, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    # Apply replacements per turn
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        if "text" in turn:
            turn["text"] = replace_masks_in_turn(turn["text"], values)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dialogue, f, ensure_ascii=False, indent=2)


# -------------------------
# Entry
# -------------------------

def main():
    files = list(DATA_DIR.rglob("*.json"))
    files = files[:20]

    for path in tqdm(files, desc="Filling masked fields"):
        try:
            process_dialogue(path)
        except Exception as e:
            print(f"⚠️ Failed {path.name}: {e}")

    print("✅ All dialogues processed")

    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)


if __name__ == "__main__":
    main()
