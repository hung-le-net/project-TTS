import soundfile as sf
import numpy as np
from datasets import load_dataset
from pathlib import Path

# ---------- CONFIG ----------
DATASET_NAME = "sdialog/voices-libritts"
SPLIT = "train"

OUTPUT_ROOT = Path("data/reference_voices")
AGENT_DIR = OUTPUT_ROOT / "agent"
CUSTOMER_DIR = OUTPUT_ROOT / "customer"

AGENT_DIR.mkdir(parents=True, exist_ok=True)
CUSTOMER_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SECONDS = 10
SAMPLE_RATE = 24000  # LibriTTS standard

N_AGENT = 20
N_CUSTOMER = 30

# ---------- HELPERS ----------
def save_first_10s(audio_array, sr, out_path):
    max_samples = TARGET_SECONDS * sr
    trimmed = audio_array[:max_samples]
    sf.write(out_path, trimmed, sr)

# ---------- MAIN ----------
def main():
    print("Loading dataset (streaming=False)...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    agent_count = 0
    customer_count = 0

    for idx, item in enumerate(ds):
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]

        # Safety check
        if sr != SAMPLE_RATE:
            audio = audio[: int(len(audio) * SAMPLE_RATE / sr)]
            sr = SAMPLE_RATE

        # Heuristic role assignment (temporary)
        if agent_count < N_AGENT:
            out_path = AGENT_DIR / f"agent_{agent_count:02d}.wav"
            save_first_10s(audio, sr, out_path)
            agent_count += 1

        elif customer_count < N_CUSTOMER:
            out_path = CUSTOMER_DIR / f"customer_{customer_count:02d}.wav"
            save_first_10s(audio, sr, out_path)
            customer_count += 1

        if agent_count >= N_AGENT and customer_count >= N_CUSTOMER:
            break

    print(f"✅ Saved {agent_count} agent voices")
    print(f"✅ Saved {customer_count} customer voices")

if __name__ == "__main__":
    main()
