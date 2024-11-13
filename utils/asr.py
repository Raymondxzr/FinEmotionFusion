import os
import json
import whisper

def transcribe_audio_files(audio_dir, output_json_path, model_name="base"):
    """
    Transcribes audio files from a directory and saves the transcriptions into a JSON file.

    Args:
        audio_dir (str): Directory containing audio files to be transcribed.
        output_json_path (str): Path where the output JSON file will be saved.
        model_name (str): Whisper model name to use for transcription (e.g., "base", "small", "medium", "large").
    """
    # Load the Whisper model
    model = whisper.load_model(model_name)
    
    # List all audio files in the directory
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".wav", ".mp3"))]
    
    # Initialize a dictionary for storing transcriptions
    transcripts = {}
    
    # Transcribe each audio file
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        print(f"Transcribing {audio_path}...")
        
        # Perform transcription
        result = model.transcribe(audio_path)
        
        # Save the transcription using the audio file name (without extension) as the key
        file_key = os.path.splitext(audio_file)[0]
        transcripts[file_key] = result["text"]
    
    # Save the transcriptions to a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(transcripts, json_file, indent=4)
    
    print(f"Transcriptions saved to {output_json_path}")

# Usage example
audio_dir = "data/audio"
output_json_path = "data/transcripts.json"
transcribe_audio_files(audio_dir, output_json_path)
