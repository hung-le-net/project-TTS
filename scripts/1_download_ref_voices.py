import random
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio

# ---------- CONFIG ----------
DATASET_NAME = "sdialog/voices-libritts"
SPLIT = "train"

OUTPUT_ROOT = Path("data/reference_voices")
AGENT_DIR = OUTPUT_ROOT / "agent"
CUSTOMER_DIR = OUTPUT_ROOT / "customer"

AGENT_DIR.mkdir(parents=True, exist_ok=True)
CUSTOMER_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SECONDS = 10
N_AGENT = 15
N_CUSTOMER = 30
RANDOM_SEED = 42

# ---------- MAIN ----------
def main():
    print("Loading dataset (audio decode disabled)...")

    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # 🔑 CRITICAL LINE — disables torchcodec
    ds = ds.cast_column("audio", Audio(decode=False))

    indices = list(range(len(ds)))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    agent_count = 0
    customer_count = 0

    for idx in indices:
        if agent_count >= N_AGENT and customer_count >= N_CUSTOMER:
            break

        item = ds[idx]
        audio_path = item["audio"]["path"]

        # Load audio manually (stable)
        audio, sr = sf.read(audio_path)

        if len(audio) < TARGET_SECONDS * sr:
            continue  # skip too-short clips

        # Decide role
        if agent_count < N_AGENT and customer_count < N_CUSTOMER:
            role = random.choice(["agent", "customer"])
        elif agent_count < N_AGENT:
            role = "agent"
        else:
            role = "customer"

        if role == "agent":
            out_path = AGENT_DIR / f"agent_{agent_count:02d}.wav"
            sf.write(out_path, audio[: TARGET_SECONDS * sr], sr)
            agent_count += 1
        else:
            out_path = CUSTOMER_DIR / f"customer_{customer_count:02d}.wav"
            sf.write(out_path, audio[: TARGET_SECONDS * sr], sr)
            customer_count += 1

    print(f"✅ Saved {agent_count} agent voices")
    print(f"✅ Saved {customer_count} customer voices")

# ---------- RUN ----------
if __name__ == "__main__":
    main()
