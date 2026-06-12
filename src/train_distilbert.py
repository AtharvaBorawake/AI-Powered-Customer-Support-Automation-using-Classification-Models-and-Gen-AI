import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

from data_loader import load_data

# -----------------------
# CONFIG (IMPROVED)
# -----------------------

MODEL_NAME = "distilbert-base-uncased"

MAX_LEN = 256  # BEST from reference paper
BATCH_SIZE = 16  # increase stability vs 8
EPOCHS = 10  # paper best range 6–7

LEARNING_RATE = 3e-5  # IMPORTANT FIX

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", DEVICE)

# -----------------------
# TEXT CLEANING
# -----------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------
# DATASET CLASS
# -----------------------

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------------
# MAIN
# -----------------------

def main():

    print("\nLoading dataset...")
    df = load_data()

    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"].tolist()
    y = df["queue"]

    # label encoding
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # -----------------------
    # SPLIT (FIXED: add validation set)
    # -----------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = TicketDataset(X_train, y_train, tokenizer)
    val_ds = TicketDataset(X_val, y_val, tokenizer)
    test_ds = TicketDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # -----------------------
    # MODEL
    # -----------------------

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(encoder.classes_)
    ).to(DEVICE)

    # -----------------------
    # CLASS WEIGHTS (GOOD FOR IMBALANCE)
    # -----------------------

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    # -----------------------
    # TRAINING
    # -----------------------

    best_f1 = 0
    patience = 2
    counter = 0

    os.makedirs("saved_models", exist_ok=True)

    print("\nTraining started...\n")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs.logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # ---------------- VALIDATION ----------------

        model.eval()

        preds, actual = [], []

        with torch.no_grad():
            for batch in val_loader:

                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                batch_preds = torch.argmax(outputs.logits, dim=1)

                preds.extend(batch_preds.cpu().numpy())
                actual.extend(labels.cpu().numpy())

        acc = accuracy_score(actual, preds)
        macro = f1_score(actual, preds, average="macro")

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss={total_loss/len(train_loader):.4f} | "
            f"Val Acc={acc:.4f} | MacroF1={macro:.4f}"
        )

        # ---------------- EARLY STOPPING ----------------

        if macro > best_f1:
            best_f1 = macro
            counter = 0

            model.save_pretrained("saved_models/distilbert_model")
            tokenizer.save_pretrained("saved_models/distilbert_model")

            print("Best model saved (validation F1 improved).")

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # ---------------- TEST EVALUATION ----------------

    model.eval()

    preds, actual = [], []

    with torch.no_grad():
        for batch in test_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            batch_preds = torch.argmax(outputs.logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            actual.extend(labels.cpu().numpy())

    print("\nFINAL RESULTS")
    print("Accuracy:", accuracy_score(actual, preds))
    print("Macro F1:", f1_score(actual, preds, average="macro"))

    print("\nClassification Report:\n")
    print(classification_report(actual, preds, target_names=encoder.classes_))

    # save encoder
    import joblib
    joblib.dump(encoder, "saved_models/label_encoder.pkl")

    print("\nModel saved to saved_models/distilbert_model")

# -----------------------

if __name__ == "__main__":
    main()