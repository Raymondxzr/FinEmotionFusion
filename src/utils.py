import os
import json
import torch
import torchaudio
from datasets import load_dataset

def create_audio_label_json(base_dir, labels_file_path, output_json_path, directories=None):
    """
    Creates a JSON file that maps audio files to labels from a specified text file.
    
    Args:
        base_dir (str): Base directory containing subdirectories with audio files.
        labels_file_path (str): Path to the text file with labels.
        output_json_path (str): Path where the output JSON will be saved.
        directories (list, optional): List of subdirectories to process in a specific order. Defaults to ["3m", "amazon", "twitter"].
    """
    # Set default directory order if none provided
    if directories is None:
        directories = ["3m", "amazon", "twitter"]

    # Read the labels from the specified labels file
    with open(labels_file_path, "r") as file:
        labels = [line.strip() for line in file.readlines()]

    # Initialize an empty list to store the mapping of audio paths to labels
    data = []
    label_index = 0  # Index to keep track of which label to assign

    # Traverse each directory in the specified order
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        
        # List all audio files in the current directory (assuming .mp3 files)
        audio_files = [f for f in sorted(os.listdir(dir_path)) if f.endswith(".mp3")]
        
        # Map each audio file to a label
        for audio_file in audio_files:
            audio_path = os.path.join(dir_path, audio_file)
            if label_index < len(labels):
                data.append({"audio_file": audio_file, "label": labels[label_index]})
                label_index += 1
            else:
                print("Warning: More audio files than labels. Some audio files will not have labels.")
                break

    # Save the data as a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file with audio-to-label mapping has been saved to {output_json_path}")






# Function to save audio and metadata
def save_audio_with_metadata(dataset, num_samples=2, output_dir="./audio_samples_with_metadata"):
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
output_dir = "./audio_samples_with_metadata"
os.makedirs(output_dir, exist_ok=True)
# Save the first two samples
save_audio_with_metadata(ds, num_samples=8, output_dir=output_dir)


# Usage example:
base_dir = 'FinEmotionFusion/src/audio_pipeline/audio_samples_with_metadata/earnings_call'
labels_file_path = 'FinEmotionFusion/src/audio_pipeline/audio_samples_with_metadata/earnings_call/predicted_labels.txt'
output_json_path = 'FinEmotionFusion/data/audio/earnings_calls_with_labels.json'

create_audio_label_json(base_dir, labels_file_path, output_json_path)


