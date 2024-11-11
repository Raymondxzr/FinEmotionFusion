import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from datasets import load_dataset

# Load the dataset
ds = load_dataset("kuanhuggingface/PromptTTS_Emotion_Recognition_8k")

print(ds['train']['file'][0])
print(ds['train']['label'][0])
# Extract the first audio sample from the dataset
audio_data = ds['train'][0]['audio']
waveform = torch.tensor(audio_data['array'])
sample_rate = audio_data['sampling_rate']

# Print current data type of waveform
print(f"Current data type of waveform: {waveform.dtype}")

# Convert the waveform to Float and print the new data type
waveform = waveform.float()
print(f"Data type after conversion: {waveform.dtype}")

# Resample the audio to 16000 Hz if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    sample_rate = 16000

# Load the feature extractor and pre-trained Wav2Vec2 model for sequence classification
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Print the sample rate after resampling
print(f"Sample rate: {sample_rate}")

# Preprocess the audio
inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

# Perform inference
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted emotion
predicted_emotion = torch.argmax(logits, dim=-1).item()

# Print the predicted emotion label using the model's config
predicted_emotion_label = model.config.id2label[predicted_emotion]
print(f"Predicted emotion label: {predicted_emotion_label}")
