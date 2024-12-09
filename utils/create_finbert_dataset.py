import os
import numpy as np
import whisper
import pandas as pd
from tqdm import tqdm

def transcribe_audio(audio_folder, labels_file, output_csv, model_name="base"):
    """
    Transcribe audio files and align them with labels.

    Args:
        audio_folder (str): Path to the folder containing audio files in WAV format.
        labels_file (str): Path to the NumPy file containing labels.
        output_csv (str): Path to save the resulting CSV file.
        model_name (str): Whisper model to use for transcription (default: "base").
    """
    # Load Whisper model
    model = whisper.load_model(model_name)

    # Load labels
    labels = np.load(labels_file)
    if len(labels) != len(os.listdir(audio_folder)):
        raise ValueError("Number of labels does not match number of audio files!")

    # Prepare transcription data
    data = []

    # Sort files to align with label order
    audio_files = sorted(os.listdir(audio_folder))

    print("Transcribing audio files...")
    for idx, audio_file in enumerate(tqdm(audio_files)):
        audio_path = os.path.join(audio_folder, audio_file)
        
        # Ensure the file is a WAV file
        if not audio_file.endswith(".wav"):
            print(f"Skipping non-WAV file: {audio_file}")
            continue
        
        # Transcribe audio
        try:
            result = model.transcribe(audio_path)
            transcript = result["text"]
        except Exception as e:
            print(f"Error transcribing {audio_file}: {e}")
            transcript = ""

        # Align with label
        label = labels[idx]
        data.append({"text": transcript, "label": label})

    # Save to CSV
    print("Saving results to CSV...")
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Transcription results saved to {output_csv}")

# Main
if __name__ == "__main__":
    audio_folder = "data/data_1000"        # Folder containing 1000 WAV files
    labels_file = "data/labels_100.npy"   # NumPy file with labels
    output_csv = "data/finbert_dataset.csv"   # Output CSV file

    transcribe_audio(audio_folder, labels_file, output_csv)
