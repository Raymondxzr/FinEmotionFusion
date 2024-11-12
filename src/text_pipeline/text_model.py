"""
Script to generate sentence embeddings for audio transcriptions using FinBERT.
Input:
    - A JSON file containing transcriptions in the format {audio_id: transcription}.
Output:
    - A JSON file containing embeddings in the format {audio_id: embedding_vector}.
"""

import json
import torch
from transformers import BertTokenizer, BertModel
from loguru import logger


def load_transcriptions(json_path: str) -> dict:
    """
    Load transcriptions from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing transcriptions.

    Returns:
        dict: A dictionary where keys are audio IDs and values are transcriptions.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            transcriptions = json.load(f)
        logger.info(f"Loaded {len(transcriptions)} transcriptions from {json_path}.")
        return transcriptions
    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        raise


def generate_embeddings(transcriptions: dict, tokenizer: BertTokenizer, model: BertModel) -> dict:
    """
    Generate sentence embeddings for the given transcriptions using the FinBERT model.

    Args:
        transcriptions (dict): A dictionary with audio IDs as keys and transcriptions as values.
        tokenizer (BertTokenizer): Pretrained FinBERT tokenizer.
        model (BertModel): Pretrained FinBERT model.

    Returns:
        dict: A dictionary with audio IDs as keys and sentence embeddings as values.
    """
    embeddings_dict = {}

    # Extract transcriptions and IDs
    sentences = list(transcriptions.values())
    audio_ids = list(transcriptions.keys())

    # Tokenize input sentences
    try:
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        logger.info("Tokenized input sentences successfully.")
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        raise

    # Generate embeddings using FinBERT
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]
        sentence_embeddings = embeddings.mean(dim=1)  # Reduce to [batch_size, hidden_size]

    # Map embeddings back to audio IDs
    for idx, audio_id in enumerate(audio_ids):
        embeddings_dict[audio_id] = sentence_embeddings[idx].tolist()

    logger.info(f"Generated embeddings for {len(embeddings_dict)} transcriptions.")
    return embeddings_dict


def save_embeddings(embeddings: dict, output_path: str) -> None:
    """
    Save sentence embeddings to a JSON file.

    Args:
        embeddings (dict): A dictionary with audio IDs as keys and embeddings as values.
        output_path (str): Path to the output JSON file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved embeddings to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving embeddings to {output_path}: {e}")
        raise


def main():
    """
    Main function to load transcriptions, generate embeddings, and save them to a file.
    """
    # Define paths
    input_path = "data/transcript_test.json"  # Path to input transcriptions JSON file
    output_path = "data/output_test.json"    # Path to save embeddings JSON file

    logger.add("logs/embedding_generation.log", rotation="1 MB", retention="1 month", level="INFO")

    try:
        # Load FinBERT tokenizer and model
        logger.info("Loading FinBERT tokenizer and model...")
        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = BertModel.from_pretrained("yiyanghkust/finbert-tone")
        logger.info("FinBERT model and tokenizer loaded successfully.")

        # Load transcriptions
        transcriptions = load_transcriptions(input_path)

        # Generate embeddings
        embeddings = generate_embeddings(transcriptions, tokenizer, model)

        # Save embeddings
        save_embeddings(embeddings, output_path)

        logger.info("Process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
