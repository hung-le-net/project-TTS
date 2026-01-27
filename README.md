# 🚀 Project TTS

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-v3.11-blue.svg) ![Status](https://img.shields.io/badge/status-active-success.svg)

> **Description:** Using SpeechLLM to generate audio from text

---

## 📋 Table of Contents

- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📂 Project Structure

```text
project-root/
├── .venv/                           # Python virtual environment
├── data/                        
│   ├── json_data/                   # Source data in JSON format
│   ├── processed_dialogues/         # Intermediate processed text/dialogues
│   ├── reference_voices/            # Audio samples used as references
│   └── synthetic_audio/             # Generated audio files
└── scripts/                     
    ├── 1_download_ref_voices.py     # Step 1: Downloads reference audio assets
    ├── 2_sentences_and_labelling.py # Step 2: Prepares sentences and assigns labels
    └── 3_synthesize_dialogues.py    # Step 3: Generates the final synthetic audio
```

## ⚡ Getting Started

### Prerequisites

* **Python 3.11** (required)
* **CUDA** (optional, if using GPU for PyTorch)
* **CPU** (run locally)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/hung-le-net/project-TTS.git
    cd project-TTS
    ```

2.  **Set up the Virtual Environment**
    It is highly recommended to use a virtual environment to manage specific dependency versions.
    ```bash
    # Create the virtual environment with Python 3.11
    python3.11 -m venv .venv

    # Activate it
    # Windows:
    .\.venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** This project relies on `torch==2.1.2` and `TTS` (Coqui). If you encounter issues with CUDA/GPU support, please refer to the [PyTorch Get Started](https://pytorch.org/get-started/previous-versions/) page for version-specific installation commands.

## 🔨 Usage

Run the scripts in the following order to generate the audio files:

1.  **Download Reference Voices**
    ```bash
    python scripts/1_download_ref_voices.py
    ```

2.  **Process Sentences & Labels**
    ```bash
    python scripts/2_sentences_and_labelling.py
    ```

3.  **Synthesize Dialogues**
    ```bash
    python scripts/3_synthesize_dialogues.py
    ```
