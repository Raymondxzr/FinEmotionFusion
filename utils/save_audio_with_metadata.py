import os
import torch
import torchaudio
from datasets import load_dataset

# Function to save audio and metadata
def save_audio_with_metadata(dataset, num_samples=2, output_dir="./audio_samples_with_metadata"):
    audio_dir = os.path.join(output_dir, "audio")
    metadata_dir = os.path.join(output_dir, "metadata")

    # Create subdirectories if they don't exist
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    for i in range(num_samples):
        # Get data
        file_path = dataset['train'][i]['file']
        audio_data = dataset['train'][i]['audio']
        label = dataset['train'][i]['label']
        transcription = dataset['train'][i]['transcription']
        audio_id = dataset['train'][i]['id']
        
        # Prepare audio
        waveform = torch.tensor(audio_data["array"]).unsqueeze(0).float()  # Add channel dimension
        sample_rate = audio_data["sampling_rate"]

        # Save the waveform as a .wav file
        wav_path = os.path.join(output_dir, f"audio/{audio_id}.wav")
        torchaudio.save(wav_path, waveform, sample_rate)

        # Save metadata as a text file
        metadata_path = os.path.join(output_dir, f"metadata/{audio_id}_metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(f"File: {file_path}\n")
            f.write(f"ID: {audio_id}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Transcription: {transcription}\n")

        print(f"Saved {wav_path} and {metadata_path}")

# Load the dataset
ds = load_dataset("kuanhuggingface/PromptTTS_Emotion_Recognition_8k")
# Directory to save files
output_dir = "../data/audio/audio_samples_with_metadata"
os.makedirs(output_dir, exist_ok=True)
# Save the first two samples
save_audio_with_metadata(ds, num_samples=8, output_dir=output_dir)
