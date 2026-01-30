import json
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import csv

import torch
from transformers import AutoModel, AutoProcessor

from reference_voices.agent_voices import AGENT_VOICES
from reference_voices.customer_voices import CUSTOMER_VOICES


# ---------- CONFIG ----------
DIALOGUE_ROOT = Path("data/processed_dialogues")
OUTPUT_ROOT = Path("data/synthetic_audio")
META_ROOT = Path("data/metadata")

SAMPLE_RATE = 12000
PAUSE_BETWEEN_TURNS = 0.4

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

META_ROOT.mkdir(parents=True, exist_ok=True)
CSV_PATH = META_ROOT / "segments.csv"

# ---------- STYLE PROMPTS ----------


# ---------- LOAD MODEL ----------
print("Loading Qwen3-TTS...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# ---------- HELPERS ----------
def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)

@torch.no_grad()
def synthesize_turn(text, style_prompt):
    prompt = (
            f"{style_prompt}\n"
            f"Text to speak:\n{text}"
        )
    
    inputs = processor(
        text=prompt,
        return_tensors="pt"
    ).to(DEVICE)

    audio = model.generate(**inputs)
    wav = audio[0].cpu().numpy()
    return wav

# ---------- MAIN ----------
def process_all():
    dialogues = list(DIALOGUE_ROOT.rglob("*_labeled.json"))
    print(f"Found {len(dialogues)} dialogues")

    write_header = not CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow([
                "audio_path",
                "speaker",
                "time_start",
                "time_end",
                "text"
            ])

        for dialog_path in tqdm(dialogues, desc="Synthesizing"):
            rel = dialog_path.relative_to(DIALOGUE_ROOT)
            out_dir = OUTPUT_ROOT / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            out_wav = out_dir / f"{dialog_path.stem}.wav"
            if out_wav.exists():
                continue

            agent_profile = random.choice(AGENT_STYLES)
            customer_profile = random.choice(CUSTOMER_STYLES)

            with open(dialog_path) as f:
                turns = json.load(f)

            full_audio = []
            current_time = 0.0

            for turn in turns:
                speaker = turn["speaker"]
                text = turn["text"].strip()
                if not text:
                    continue

                style = (
                    agent_profile["prompt"]
                    if speaker == "agent"
                    else customer_profile["prompt"]
                )

                speaker_id = (
                    agent_profile["id"]
                    if speaker == "agent"
                    else customer_profile["id"]
                )

                wav = synthesize_turn(text, style)

                duration = len(wav) / SAMPLE_RATE
                start = current_time
                end = start + duration

                writer.writerow([
                    out_wav.name,
                    speaker_id,
                    round(start, 3),
                    round(end, 3),
                    text
                ])

                full_audio.append(wav)
                full_audio.append(silence(PAUSE_BETWEEN_TURNS))
                current_time = end + PAUSE_BETWEEN_TURNS

            if full_audio:
                sf.write(out_wav, np.concatenate(full_audio), SAMPLE_RATE)

    print("✅ Qwen3-TTS synthesis complete")

# ---------- RUN ----------
if __name__ == "__main__":
    process_all()
