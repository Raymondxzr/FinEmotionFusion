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
4. Move earnings_call folder to the top-level and then create audio-label mapping:
   ```bash
  python utils/create_audio_label_json.py
  ```

### Project Structure
  ```bash
EmotiFusion/
├── data/
│   ├── audio/               # Raw audio data
│   ├── transcripts.json         # Raw text data
│   ├── labels.json
├── models/
│   ├── finbert/             # Saved models
├── src/
│   ├── audio_pipeline/
│   │   ├── preprocess_audio.py   # Preprocesses raw audio data
│   │   ├── audio_model.py        # Model for processing audio data
│   │   └── ...
│   ├── data_pipeline/
│   │   ├── train_cnn.py          # Script for training CNN for data generation
│   ├── text_pipeline/
│   │   ├── preprocess_text.py    # Preprocesses raw text data
│   │   ├── text_model.py         # Model for processing text data
│   │   └── ...
│   ├── fusion/
│   │   ├── alignment.py          # Aligning two modalities using cross-attention
│   │   ├── classifier.py         # Final classifier
│   │   ├── fusion.py             # Multimodal
│   │   └── ...
├── tests/
│   ├── test_audio_pipeline.py    # Tests for audio pipeline
│   ├── test_baseline.py          # Baseline model tests
│   ├── test_text_pipeline.py     # Tests for text pipeline
│   ├── test_fusion.py            # Tests for fusion model
│   └── ...
├── utils/
│   ├── asr.py                              # Converting audio to text
│   ├── create_audio_label_json.py          # Mapping from audio to label
│   ├── save_audio_with_metadata.py         # Tests with 8k data
│   ├── util.py                             # Utility functions
│   └── ...
├── requirements.txt
├── README.md
└── main.py                                 # Main.py
```

