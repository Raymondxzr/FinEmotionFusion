import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        """
        Custom Dataset for text classification.

        Args:
            dataframe (pd.DataFrame): Dataframe containing 'text' and 'label' columns.
            tokenizer (BertTokenizer): Pretrained tokenizer.
            max_length (int): Maximum sequence length.
        """
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Training Function
def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        # Training Loop
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation Loop
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average="weighted")
        print(f"Epoch {epoch+1} Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    return model

# Main Script
if __name__ == "__main__":
    # Paths
    data_path = "data/finbert_dataset.csv"  
    output_dir = "model/finbert-finetuned"

    # Load Data
    df = pd.read_csv(data_path)
    df = df[df['text'].apply(lambda x: isinstance(x, str))]

    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    invalid_entries = val_df[val_df['text'].apply(lambda x: not isinstance(x, str))]
    print(invalid_entries)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

    # Datasets and DataLoaders
    train_dataset = TextDataset(train_df, tokenizer)
    val_dataset = TextDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 8
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=num_labels, ignore_mismatched_sizes=True)

    # Train
    epochs = 4
    learning_rate = 2e-5
    model = train_model(model, train_loader, val_loader, epochs, learning_rate, device)

    # Save the Fine-Tuned Model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
