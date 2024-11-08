# FinEmotionFusion

FinEmotionFusion is an emotion detection system designed for financial phone calls, leveraging early fusion of audio and text modalities to enhance accuracy and context-awareness.

## Getting Started

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed.

### Installation

1. Create a new Conda environment:
  ```bash
  conda create -n fine python=3.10
  ```
2. Activate the environment:
  ```bash
  conda activate fine
  ```
3. Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Project Structure
  ```bash
EmotiFusion/
├── data/
│   ├── audio/               # Raw audio data
│   ├── transcripts/         # Raw text data
├── models/
│   ├── finbert/             # Saved models
├── src/
│   ├── audio_pipeline/
│   │   ├── preprocess_audio.py   # Preprocesses raw audio data
│   │   ├── audio_model.py        # Model for processing audio data
│   │   └── ...
│   ├── text_pipeline/
│   │   ├── preprocess_text.py    # Preprocesses raw text data
│   │   ├── text_model.py         # Model for processing text data
│   │   └── ...
│   ├── early_fusion/
│   │   └── ...                   # Code for early fusion of audio and text modalities
│   └── utils.py                  # Utility functions
├── tests/
│   ├── test_audio_pipeline.py    # Tests for audio pipeline
│   ├── test_baseline.py          # Baseline model tests
│   ├── test_text_pipeline.py     # Tests for text pipeline
│   └── test_fusion.py            # Tests for fusion model
├── requirements.txt
└── README.md
```

