import os
import json


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

# Usage example:
base_dir = '../src/audio_pipeline/audio_samples_with_metadata/earnings_call'
labels_file_path = '../src/audio_pipeline/audio_samples_with_metadata/earnings_call/predicted_labels.txt'
output_json_path = '../data/audio/earnings_calls_with_labels.json'

create_audio_label_json(base_dir, labels_file_path, output_json_path)
