import json
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

import torch
from TTS.api import TTS

# ---------- CONFIG ----------
DIALOGUE_ROOT = Path("data/processed_dialogues")
VOICE_ROOT = Path("data/reference_voices")
OUTPUT_ROOT = Path("data/synthetic_audio")

AGENT_VOICES = list((VOICE_ROOT / "agent").glob("*.wav"))
CUSTOMER_VOICES = list((VOICE_ROOT / "customer").glob("*.wav"))

SAMPLE_RATE = 24000
PAUSE_BETWEEN_TURNS = 0.4  # seconds of silence

XTTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD MODEL ----------
print("Loading XTTS-v2...")
tts = TTS(
    model_name=XTTS_MODEL_ID,
    progress_bar=False
).to(DEVICE)


# ---------- HELPERS ----------
def silence(duration_sec):
    return np.zeros(int(duration_sec * SAMPLE_RATE), dtype=np.float32)

def synthesize_turn(text, voice_path):
    wav = tts.tts(
        text=text,
        speaker_wav=str(voice_path),
        language="en"
    )
    return wav

# ---------- MAIN ----------
def process_all():
    dialogues = list(DIALOGUE_ROOT.rglob("*_labeled.json"))
    print(f"Found {len(dialogues)} dialogues")

    for dialog_path in tqdm(dialogues, desc="Synthesizing"):
        rel = dialog_path.relative_to(DIALOGUE_ROOT)
        out_dir = OUTPUT_ROOT / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_wav = out_dir / f"{dialog_path.stem}.wav"
        if out_wav.exists():
            continue

        # Randomly pick voices for THIS dialogue
        agent_voice = random.choice(AGENT_VOICES)
        customer_voice = random.choice(CUSTOMER_VOICES)

        with open(dialog_path) as f:
            turns = json.load(f)

        full_audio = []

        for turn in turns:
            speaker = turn["speaker"]
            text = turn["text"].strip()

            if not text:
                continue

            voice = agent_voice if speaker == "agent" else customer_voice
            wav = synthesize_turn(text, voice)

            full_audio.append(wav)
            full_audio.append(silence(PAUSE_BETWEEN_TURNS))

        if full_audio:
            dialogue_audio = np.concatenate(full_audio)
            sf.write(out_wav, dialogue_audio, SAMPLE_RATE)

    print("✅ Synthesis complete")

# ---------- RUN ----------
if __name__ == "__main__":
    process_all()
