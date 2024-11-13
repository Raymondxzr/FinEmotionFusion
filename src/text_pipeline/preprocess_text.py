import os
import json
from transformers import BertTokenizer
from tqdm import tqdm

class TextPreprocessor:
    def __init__(self, transcripts_path, tokenizer_model="yiyanghkust/finbert-tone", max_length=128, batch_size=4):
        """
        Initialize the TextPreprocessor.

        Args:
            transcripts_path (str): Path to the JSON file containing transcriptions.
            tokenizer_model (str): Pretrained tokenizer model.
            max_length (int): Maximum token length for padding/truncation.
            batch_size (int): Batch size for tokenizing the text.
        """
        self.transcripts_path = transcripts_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length
        self.batch_size = batch_size

    def load_transcripts(self):
        """
        Load transcriptions from the JSON file.

        Returns:
            dict: Dictionary of audio file names and their corresponding text transcriptions.
        """
        if not os.path.exists(self.transcripts_path):
            raise FileNotFoundError(f"Transcripts file not found at {self.transcripts_path}")
        
        with open(self.transcripts_path, "r") as file:
            transcripts = json.load(file)
        
        return transcripts

    def tokenize_batch(self, batch_texts):
        """
        Tokenize a batch of text.

        Args:
            batch_texts (list): List of text strings to tokenize.

        Returns:
            dict: Tokenized inputs compatible with transformer models.
        """
        tokenized = self.tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return tokenized

    def preprocess(self):
        """
        Preprocess all transcriptions.

        Returns:
            dict: Dictionary containing tokenized inputs for all transcriptions.
        """
        transcripts = self.load_transcripts()
        all_keys = list(transcripts.keys())
        all_texts = list(transcripts.values())

        tokenized_data = {}
        
        for i in tqdm(range(0, len(all_texts), self.batch_size), desc="Tokenizing text batches"):
            batch_keys = all_keys[i:i + self.batch_size]
            batch_texts = all_texts[i:i + self.batch_size]
            tokenized_inputs = self.tokenize_batch(batch_texts)
            
            # Map back to audio file names
            for j, key in enumerate(batch_keys):
                tokenized_data[key] = {
                    "input_ids": tokenized_inputs["input_ids"][j].tolist(),
                    "attention_mask": tokenized_inputs["attention_mask"][j].tolist(),
                }

        return tokenized_data

    def save_tokenized_data(self, output_path):
        """
        Save the tokenized data to a JSON file.

        Args:
            output_path (str): Path to save the tokenized data.
        """
        tokenized_data = self.preprocess()
        
        with open(output_path, "w") as file:
            json.dump(tokenized_data, file, indent=4)
        
        print(f"Tokenized data saved to {output_path}")


# Usage Example
if __name__ == "__main__":
    transcripts_path = "data/transcripts.json"
    output_path = "data/tokenized_transcripts.json"

    text_preprocessor = TextPreprocessor(transcripts_path)
    text_preprocessor.save_tokenized_data(output_path)
