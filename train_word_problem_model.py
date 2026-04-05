"""
train_word_problem_model.py
----------------------------
Fine-tunes T5-small on the (schema -> sentence) pairs produced by
word_problem_generator.py.

Usage:
    python3 train_word_problem_model.py

Outputs:
    word_problem_model/   <- directory containing the fine-tuned model + tokenizer

Requirements:
    pip install torch transformers datasets sentencepiece
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from word_problem_generator import generate_training_pairs, save_pairs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAIRS_PATH   = os.path.join(os.path.dirname(__file__), "word_problem_pairs.json")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "word_problem_model")
BASE_MODEL   = "t5-small"
MAX_INPUT    = 128
MAX_TARGET   = 128
BATCH_SIZE   = 8
EPOCHS       = 5
LEARNING_RATE = 3e-4
TRAIN_SPLIT  = 0.9
SEED         = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WordProblemDataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, max_input: int, max_target: int):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_input  = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        inp = self.tokenizer(
            pair["schema"],
            max_length=self.max_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            pair["sentence"],
            max_length=self.max_target,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      inp["input_ids"].squeeze(),
            "attention_mask": inp["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print(f"Loading / generating training pairs...")
    if os.path.exists(PAIRS_PATH):
        with open(PAIRS_PATH) as f:
            pairs = json.load(f)
        print(f"  Loaded {len(pairs)} pairs from {PAIRS_PATH}")
    else:
        pairs = generate_training_pairs(n=500, seed=SEED)
        save_pairs(pairs, PAIRS_PATH)

    import random
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    split = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]
    print(f"  Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    print(f"Loading tokenizer and model: {BASE_MODEL}")
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    model     = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    train_ds = WordProblemDataset(train_pairs, tokenizer, MAX_INPUT, MAX_TARGET)
    val_ds   = WordProblemDataset(val_pairs,   tokenizer, MAX_INPUT, MAX_TARGET)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_dl:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_dl)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()

        avg_val = total_val_loss / len(val_dl)
        print(f"  Epoch {epoch}/{EPOCHS}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"    Saved best model to {MODEL_DIR}  (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Re-train even if a saved model already exists in word_problem_model/"
    )
    args = parser.parse_args()

    if os.path.isdir(MODEL_DIR) and not args.force:
        print(f"Model already exists at {MODEL_DIR}")
        print("Skipping training. Use --force to retrain.")
    else:
        train()
