# train_lstm.py

import os
import re
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.utils.class_weight import compute_class_weight
# ---------------------------------------------------
# Import your existing data loader
# ---------------------------------------------------

from data_loader import load_data

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

MAX_WORDS = 40000
MAX_LEN = 200

EMBED_DIM = 256
HIDDEN_DIM = 256

BATCH_SIZE = 64
EPOCHS = 40

LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", DEVICE)

# ---------------------------------------------------
# Text Cleaning
# ---------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ---------------------------------------------------
# Dataset Class
# ---------------------------------------------------

class TicketDataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = torch.tensor(
            sequences,
            dtype=torch.long
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        return self.sequences[idx], self.labels[idx]

# ---------------------------------------------------
# BiLSTM Model
# ---------------------------------------------------

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim
    ):

        super().__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # Embedding Dropout
        self.embed_dropout = nn.Dropout1d(0.2)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Regularization
        self.dropout = nn.Dropout(0.3)

        # Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            batch_first=True,
            dropout=0.2
        )

        # Layer Normalization
        self.norm = nn.LayerNorm(
            hidden_dim * 4 + embedding_dim
        )

        # Classifier
        self.fc = nn.Sequential(

            nn.Linear(hidden_dim * 4 + embedding_dim, 256),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(256, 128),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(128, output_dim)

        )

    def forward(self, x):

        # Padding mask
        mask = (x != 0)

        # Embedding
        embeddings = self.embedding(x)

        # Residual embedding representation
        embedding_mean = (
            embeddings * mask.unsqueeze(-1)
        ).sum(dim=1)

        embedding_mean = embedding_mean / (
            mask.sum(dim=1, keepdim=True).clamp(min=1)
        )

        # Embedding Dropout
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.embed_dropout(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        # BiLSTM
        output, _ = self.lstm(embeddings)

        # Multi-head Self Attention
        attn_output, _ = self.attention(
            output,
            output,
            output,
            key_padding_mask=~mask
        )

        # Masked Attention Mean Pool
        attention_pool = (
            attn_output * mask.unsqueeze(-1)
        ).sum(dim=1)

        attention_pool = attention_pool / (
            mask.sum(dim=1, keepdim=True).clamp(min=1)
        )

        # Masked Max Pool
        masked_output = output.masked_fill(
            ~mask.unsqueeze(-1),
            -1e9
        )

        max_pool, _ = torch.max(
            masked_output,
            dim=1
        )

        # Final Representation
        pooled = torch.cat(
            (
                attention_pool,
                max_pool,
                embedding_mean
            ),
            dim=1
        )

        pooled = self.norm(pooled)

        pooled = self.dropout(pooled)

        logits = self.fc(pooled)

        return logits
        
# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df = load_data()

    print(df.shape)

    print("Cleaning text...")

    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]

    y = df["queue"]

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    print("Train Test Split...")

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("Building tokenizer...")

    tokenizer = Tokenizer(

        num_words=MAX_WORDS,

        oov_token="<OOV>"

    )

    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)

    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(

        X_train,

        maxlen=MAX_LEN,

        padding="post",

        truncating="post"

    )

    X_test = pad_sequences(

        X_test,

        maxlen=MAX_LEN,

        padding="post",

        truncating="post"

    )

    print("Creating DataLoaders...")

    train_dataset = TicketDataset(

        X_train,

        y_train

    )

    test_dataset = TicketDataset(

        X_test,

        y_test

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE

    )

    vocab_size = min(

        MAX_WORDS,

        len(tokenizer.word_index) + 1

    )

    model = BiLSTMClassifier(

        vocab_size=vocab_size,

        embedding_dim=EMBED_DIM,

        hidden_dim=HIDDEN_DIM,

        output_dim=len(encoder.classes_)

    ).to(DEVICE)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )
    
   # ---------------------------------------------------
    # Training
    # ---------------------------------------------------

    best_macro = 0.0

    patience = 8

    counter = 0

    os.makedirs("saved_models", exist_ok=True)

    print("=" * 60)
    print("Training Started")
    print("=" * 60)

    for epoch in range(EPOCHS):

        # ------------------------
        # Train
        # ------------------------

        model.train()

        total_loss = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        # ------------------------
        # Validation
        # ------------------------

        model.eval()

        predictions = []

        actual = []

        with torch.no_grad():

            for inputs, labels in test_loader:

                inputs = inputs.to(DEVICE)

                outputs = model(inputs)

                preds = torch.argmax(outputs, dim=1)

                predictions.extend(
                    preds.cpu().numpy()
                )

                actual.extend(
                    labels.numpy()
                )

        acc = accuracy_score(
            actual,
            predictions
        )

        macro = f1_score(
            actual,
            predictions,
            average="macro"
        )

        scheduler.step(macro)

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
            f" | Loss={total_loss/len(train_loader):.4f}"
            f" | Accuracy={acc:.4f}"
            f" | MacroF1={macro:.4f}"
        )

        # ------------------------
        # Save best model
        # ------------------------

        if macro > best_macro:

            best_macro = macro

            counter = 0

            torch.save(
                model.state_dict(),
                "saved_models/bilstm_best.pth"
            )

            print("Best model updated.")

        else:

            counter += 1

            print(
                f"No improvement ({counter}/{patience})"
            )

            if counter >= patience:

                print("\nEarly stopping triggered.")

                break


    # ---------------------------------------------------
    # Load Best Model
    # ---------------------------------------------------

    print("\nLoading best model...")

    model.load_state_dict(

        torch.load(

            "saved_models/bilstm_best.pth",

            map_location=DEVICE

        )

    )

    model.eval()

    predictions = []

    actual = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            actual.extend(
                labels.numpy()
            )

    acc = accuracy_score(
        actual,
        predictions
    )

    macro = f1_score(
        actual,
        predictions,
        average="macro"
    )

    print()

    print("Final Accuracy :", acc)

    print("Final Macro F1 :", macro)

    print()

    print(

        classification_report(

            actual,

            predictions,

            target_names=encoder.classes_

        )

    )

    joblib.dump(

        tokenizer,

        "saved_models/tokenizer.pkl"

    )

    joblib.dump(

        encoder,

        "saved_models/label_encoder.pkl"

    )

    print()

    print("Best model, tokenizer and label encoder saved successfully.") 


if __name__ == "__main__":

    main()