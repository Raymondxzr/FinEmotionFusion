import librosa
import numpy as np
import os
import torch
from tqdm import tqdm

def extract_audio_features(file_path, sr=22050):
    """
    Extracts 180 spectral features from an audio file: 
    - 40 Mel-frequency cepstral coefficients (MFCCs)
    - 12 chromagram features
    - 128 Mel-scaled spectrogram features

    Parameters:
    - file_path (str): Path to the audio file
    - sr (int): Sampling rate for the audio file

    Returns:
    - np.array: A 1D array with 180 features
    """
    # Load audio file
    y, _ = librosa.load(file_path, sr=sr)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)  # Take mean along time axis

    # Extract Chromagram features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # Take mean along time axis

    # Extract Mel-scaled spectrogram features
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)  # Take mean along time axis

    # Concatenate all features into a single feature vector
    feature_vector = np.hstack([mfcc_mean, chroma_mean, mel_spectrogram_mean])

    return np.array(feature_vector)


def preprocess_audio_data_to_tensor(data_dir='data/audio'):
    """
    Processes all .wav files in the specified directory, extracting features from each file 
    and combining them into a single tensor dataset.

    Parameters:
    - data_dir (str): Root directory where audio folders for each actor are stored.

    Returns:
    - torch.Tensor: A tensor of shape [number of recordings, 180]
    - list: List of labels indicating the actor folder for each recording
    """
    all_features = []
    labels = []
    
    # Iterate through each actor folder
    for actor_dir in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor_dir)
        if os.path.isdir(actor_path):  # Ensure it is a directory
            # Iterate through each .wav file in the actor's folder
            for wav_file in tqdm(sorted(os.listdir(actor_path)), desc=f"Processing {actor_dir}"):
                file_path = os.path.join(actor_path, wav_file)
                
                # Extract features and convert to a tensor
                features = extract_audio_features(file_path)
                all_features.append(np.array(features))
                
                # Add the actor directory as a label
                labeling_info = np.array([int(temp[-1]) for temp in wav_file[:-4].split('-')])
                labels.append(labeling_info)
                
    # Convert the list of features to a tensor
    features_tensor = torch.tensor(all_features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return features_tensor, labels_tensor

# Example usage
data_dir = '../../data/audio/cnn_training_data'  # Path to the root audio folder
features_tensor, labels_tensor = preprocess_audio_data_to_tensor(data_dir)
print("Extracted tensor shape:", features_tensor.shape)  # Expected shape: [number of recordings, 180]

# If you want to save the tensor for later usage
torch.save(features_tensor, '../../data/audio/audio_features_tensor.pt')
torch.save(labels_tensor, '../../data/audio/audio_labels_tensor.pt')