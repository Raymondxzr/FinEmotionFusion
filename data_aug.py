import os
import pandas as pd
import json
import re
from transformers import pipeline, MarianMTModel, MarianTokenizer
from gtts import gTTS
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()")

# ------------- Configuration -------------

# Path to the input CSV file
CSV_PATH = 'data/finbert_dataset.csv'

# Directory to save augmented audio files
AUDIO_DIR = 'data/audio'

# Output JSON files
AUDIO_TO_LABEL_JSON = 'data/aug_labels.json'
AUDIO_TO_TEXT_JSON = 'data/aug_transcripts.json'

# Languages for back-translation
SRC_LANG = 'en'
MID_LANG = 'fr'  # French

# Number of augmentations per sample
AUGMENTATIONS_PER_SAMPLE = 2  # 1: Paraphrasing, 2: Back-Translation

# Ensure the audio directory exists
# os.makedirs(AUDIO_DIR, exist_ok=True)

# ------------- Utility Functions -------------

def sanitize_filename(name):
    return re.sub(r'\W+', '_', name)

def paraphrase_text(paraphraser, text):
    try:
        paraphrased = paraphraser(f"{text}", max_length=512, num_return_sequences=1)[0]['generated_text']
        return paraphrased
    except Exception as e:
        print(f"Error in paraphrasing: {e}")
        return text  # Return original text if paraphrasing fails

def back_translate(back_translator, text):
    
    try:
        translated = back_translator(text, max_length=512, num_return_sequences=1)[0]['generated_text']
        return translated
    except Exception as e:
        print(f"Error in back-translation: {e}")
        return text  # Return original text if back-translation fails

def synthesize_speech(text, output_path):
    """Convert text to speech and save it to the given path."""
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(output_path)
        return True  # Return True to indicate success
    except Exception as e:
        return False  # Return False to indicate failure


# ------------- Initialize Models -------------

print("Initializing models...")

# Initialize Paraphrasing Pipeline
paraphraser = pipeline('text2text-generation', model='t5-base', tokenizer='t5-base')

# Initialize Back-Translation Pipelines
# English to French
mt_en_fr_tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{SRC_LANG}-{MID_LANG}')
mt_en_fr_model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{SRC_LANG}-{MID_LANG}')
back_translator_en_fr = pipeline('translation_en_to_fr', model=mt_en_fr_model, tokenizer=mt_en_fr_tokenizer)

# French to English
mt_fr_en_tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{MID_LANG}-{SRC_LANG}')
mt_fr_en_model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{MID_LANG}-{SRC_LANG}')
back_translator_fr_en = pipeline('translation_fr_to_en', model=mt_fr_en_model, tokenizer=mt_fr_en_tokenizer)


# Combined Back-Translator
def back_translate_pipeline(text):
        
    try:
        # Translate English to French
        translated = back_translator_en_fr(text, max_length=512, num_return_sequences=1)[0]['translation_text']
        # Translate French back to English
        back_translated = back_translator_fr_en(translated, max_length=512, num_return_sequences=1)[0]['translation_text']
        return back_translated
    except Exception as e:
        print(f"Error in back-translation pipeline: {e}")
        return text  # Return original text if any step fails

print("Models initialized successfully.")

# ------------- Read and Process CSV -------------

# Read the CSV file
try:
    df = pd.read_csv(CSV_PATH).head(20)
except FileNotFoundError:
    print(f"Error: The file '{CSV_PATH}' was not found.")
    exit(1)
except Exception as e:
    print(f"Error reading '{CSV_PATH}': {e}")
    exit(1)

# Verify required columns
required_columns = {'text', 'label'}
if not required_columns.issubset(df.columns):
    print(f"Error: CSV file must contain the following columns: {required_columns}")
    exit(1)

# Initialize JSON mappings
audio_to_label = {}
audio_to_text = {}

# ------------- Data Augmentation and Audio Generation -------------

print("Starting data augmentation and audio generation...")

for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
    text = row['text']
    label = row['label']
    
    # Sanitize speaker name for filenames
    # speaker_sanitized = sanitize_filename(speaker)
    
    # Row index for naming (start at 1 for readability)
    row_index = idx + 1
    
    # Generate Augmentation 1: Paraphrasing
    paraphrased_text = paraphrase_text(paraphraser, text)
    
    # Define audio filename
    audio_filename_p = f"augmented_{row_index}_1.wav"
    audio_path_p = os.path.join(AUDIO_DIR, audio_filename_p)
        
    # Update JSON mappings
    if synthesize_speech(paraphrased_text, audio_path_p):
        # Update JSON mappings only if synthesis is successful
        audio_to_label[audio_filename_p] = label
        audio_to_text[audio_filename_p] = paraphrased_text

    
    # Generate Augmentation 2: Back-Translation
    back_translated_text = back_translate_pipeline(text)
    
    # Define audio filename
    audio_filename_bt = f"augmented_{row_index}_2.wav"
    audio_path_bt = os.path.join(AUDIO_DIR, audio_filename_bt)
    
    # Synthesize speech
    if synthesize_speech(back_translated_text, audio_path_bt):
        # Update JSON mappings
        audio_to_label[audio_filename_bt] = label
        audio_to_text[audio_filename_bt] = back_translated_text

print("Data augmentation and audio generation completed.")

# ------------- Save JSON Mappings -------------

print("Saving JSON mappings...")

with open(AUDIO_TO_LABEL_JSON, 'w') as f_label:
    json.dump(audio_to_label, f_label, indent=4)

with open(AUDIO_TO_TEXT_JSON, 'w') as f_text:
    json.dump(audio_to_text, f_text, indent=4)

print(f"JSON files saved:\n- {AUDIO_TO_LABEL_JSON}\n- {AUDIO_TO_TEXT_JSON}")
print("All tasks completed successfully.")