import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from audio_dataloader import *

class Wav2Vec2FeatureExtractorOnly:
    def __init__(self, model_name="facebook/wav2vec2-base"):
        # Load the Wav2Vec2 model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def extract_features_batch(self, dataloader):
        feature_vectors = []

        # Process each batch
        for batch in dataloader:
            batch = batch.squeeze(1)
            with torch.no_grad():
                # Pass the batch through the model
                features = self.model(batch).last_hidden_state
                # Pool features and add to the list
                pooled_features = features.mean(dim=1)  # Mean pooling
                feature_vectors.extend(pooled_features.cpu().numpy())  # Append as numpy arrays

        return feature_vectors

if __name__ == '__main__':
    # Initialize feature extractor and dataset
    feature_extractor_only = Wav2Vec2FeatureExtractorOnly()
    audio_dir_path = 'audio_samples_with_metadata/earnings_call/amazon'
    dataset = AudioDataset(audio_dir_path, feature_extractor_only.feature_extractor)

    # Create a DataLoader with batch size 4 and num_workers=0
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)  # Set num_workers=0

    # Extract features in batches
    feature_vectors = feature_extractor_only.extract_features_batch(dataloader)
    print("Extracted feature vectors:", feature_vectors)