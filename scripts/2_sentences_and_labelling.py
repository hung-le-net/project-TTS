import json
import re
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# ---------- CONFIG ----------
INPUT_ROOT = Path("data/json_data")
OUTPUT_ROOT = Path("data/processed_dialogues")

AGENT_KEYWORDS = [
    "verify",
    "verify your",
    "identifying information",
    "pull up",
    "records",
    "recorded",
    "system",
    "database",
    "notate",
    "security purpose",
    "secure line",
    "reference number",
    "confirmation number",
    "standard procedure",
    "transfer you"
    "inconvenience",
    "assist",
    "assistance",
    "brief hold",
    "hold",
    "place you on",
    "department",
    "escalate",
    "anything else",
    "reference",
    "appreciate",
    "I can",
    "I will",
    "Just so you know",
    "press",
    "enter",
    "your",
    "download",
    "stay on the line",
    "apply",
    "patient",
    "if you do",
    "you may",
    "as soon as",
    "mobile carriers",
    "speaking",
    "I'm one of the",
    "easier for you",
    "help you",
    "are you interested in",
    "you wouldn't need to",
    "if you want to",
    "no worry",
    "how are you",
    "do you currently have",
    "this is",
    "what you want",
    "if you have received",
    "qualified",
    "thank you for calling",
    "behalf of",
    "appreciate your patience",
    "please choose",
    "you have the right",
    "are busy"
]

CUSTOMER_KEYWORDS = [
    "refund",
    "charged",
    "charged me",
    "cancel",
    "ridiculous",
    "frustrating",
    "waste",
    "not working",
    "ordered",
    "supposed",
    "tried",
    "already tried",
    "spoke to",
    "speak to",
    "manager",
    "told me",
    "promised",
    "said that",
    "why is",
    "how come",
    "bill",
    "confused",
    "ago",
    "days",
    "weeks",
    "last time",
    "supposed to",
    "broken",
    "I want",
    "I need",
    "I'm interested in",
    "the right person",
    "I think",
    "like a",
    "on your website",
    "I like this",
    "myself",
    "I have here with me",
    "I'm looking for",
    "on the other line",
    "I'm getting better",
    "my bill",
    "my account",
    "my card",
    "my money",

]

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

ensure_nltk()

# ---------- HELPERS ----------
def split_sentences(text: str):
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 1]


def keyword_score(text, keywords):
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)

def label_sentences(sentences):
    labeled = []
    previous_speaker = "agent"

    for i, sent in enumerate(sentences):
        a_score = keyword_score(sent, AGENT_KEYWORDS)
        c_score = keyword_score(sent, CUSTOMER_KEYWORDS)

        if i == 0:
            speaker = "agent"
        elif a_score > c_score:
            speaker = "agent"
        elif c_score > a_score:
            speaker = "customer"
        else:
            speaker = "customer" if previous_speaker == "agent" else "agent"

        labeled.append({
            "turn_id": i,
            "speaker": speaker,
            "text": sent
        })

        previous_speaker = speaker

    return labeled

# ---------- MAIN ----------
def process_all():
    files = list(INPUT_ROOT.rglob("*.json"))
    print(f"Found {len(files)} JSON files")

    for path in tqdm(files, desc="Processing dialogues"):
        rel = path.relative_to(INPUT_ROOT)
        out_dir = OUTPUT_ROOT / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"{path.stem}_labeled.json"
        if out_file.exists():
            continue

        try:
            with open(path) as f:
                data = json.load(f)

            text = data.get("text", "").strip()
            if not text:
                continue

            sentences = split_sentences(text)
            labeled = label_sentences(sentences)

            with open(out_file, "w") as f:
                json.dump(labeled, f, indent=2)

        except Exception as e:
            print(f"⚠️ Failed {path}: {e}")

    print("✅ Done")

if __name__ == "__main__":
    process_all()
