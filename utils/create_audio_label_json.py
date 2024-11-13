"""
Script moving audio files, and creating JSON mappings of audio files to labels. 
"""
import os
import json
import shutil
import re

def custom_sort_key(filename):
    # Use regex to extract the two numbers from the filename
    match = re.search(r'_(\d+)_(\d+)', filename)
    if match:
        number1 = int(match.group(1))  # First number after the first underscore
        number2 = int(match.group(2))  # Second number after the second underscore
        return (number1, number2)
    else: 
        print(filename)
    return (float('inf'), float('inf'))  # Place files without matching pattern at the end


def move_audio_files(src_dir, dest_dir, directories=None, audio_extensions=(".wav", ".mp3")):
    """
    Moves all audio files from specified subdirectories within src_dir to dest_dir in a specific order.

    Args:
        src_dir (str): Source directory containing subdirectories with audio files.
        dest_dir (str): Destination directory where audio files will be moved.
        directories (list, optional): List of subdirectories to process in a specific order. 
                                      Defaults to None, in which case all subdirectories are processed in alphabetical order.
        audio_extensions (tuple): Tuple of audio file extensions to move (default: (".wav", ".mp3")).
    """
    # Set default directory order if none is provided
    if directories is None:
        directories = sorted([d for d in (os.listdir(src_dir)) if os.path.isdir(os.path.join(src_dir, d))])

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Traverse each directory in the specified order
    for dir_name in directories:
        dir_path = os.path.join(src_dir, dir_name)
        
        # List all audio files in the current directory
        audio_files = [f for f in os.listdir(dir_path) if f.endswith(audio_extensions)]
        
        # Move each audio file to the destination directory
        for audio_file in audio_files:
            src_file_path = os.path.join(dir_path, audio_file)
            dest_file_path = os.path.join(dest_dir, audio_file)

            # If a file with the same name exists in the destination, add a suffix to avoid overwriting
            # if os.path.exists(dest_file_path):
            #     base, ext = os.path.splitext(audio_file)
            #     counter = 1
            #     while os.path.exists(dest_file_path):
            #         dest_file_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
            #         counter += 1

            # Move the audio file
            shutil.copy(src_file_path, dest_file_path)
            print(f"Moved: {src_file_path} to {dest_file_path}")

    print(f"All audio files have been moved to {dest_dir} in the specified order.")





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
    data = {}
    label_index = 0  # Index to keep track of which label to assign

    # Traverse each directory in the specified order
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        # List all audio files in the current directory (assuming .mp3 files)
        audio_files = [f for f in os.listdir(dir_path) if f.endswith(".mp3")]
        sorted_files_list = sorted(audio_files, key=custom_sort_key)
        # print(audio_files)
        # return
        # Map each audio file to a label
        for audio_file in sorted_files_list:
            if label_index < len(labels):
                data[os.path.splitext(audio_file)[0]] = int(labels[label_index])
                # data.append({"audio_file": audio_file, "label": labels[label_index]})
                label_index += 1
            else:
                print("Warning: More audio files than labels. Some audio files will not have labels.")
                break

    # Save the data as a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file with audio-to-label mapping has been saved to {output_json_path}")

    

base_dir = '../earnings_call'
labels_file_path = '../predicted_labels.txt'
output_json_path = '../data/labels.json'
create_audio_label_json(base_dir, labels_file_path, output_json_path)


dest_dir = '../data/audio'
directories = ["3m", "amazon", "twitter"]  # Specify the directory order

move_audio_files(base_dir, dest_dir, directories=directories)