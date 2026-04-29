import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Prevent the tokenizer from deadlocking DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from helper_code.mimic_dataset import mimic_dataset
from transformer_lstm import transformer_lstm


# --------------------------------------------------------------------------- #
# Dataset wrapper for precomputed LSTM embeddings
# --------------------------------------------------------------------------- #
class lstm_precomputed_dataset(torch.utils.data.Dataset):
    """
    Wraps precomputed sequence embeddings + attention masks into a Dataset.

    Returns:
        x_dict : {'embeddings': Tensor(seq_len, 768) float16,
                  'attention_mask': Tensor(seq_len) int64}
        label  : scalar Tensor
    """
    def __init__(self, embeddings, masks, labels):
        assert len(embeddings) == len(masks) == len(labels)
        self.embeddings = embeddings    # (N, seq_len, 768)  float16
        self.masks      = masks         # (N, seq_len)
        self.labels     = labels        # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            {
                'embeddings':    self.embeddings[idx],
                'attention_mask': self.masks[idx],
            },
            self.labels[idx],
        )


def main():
    # ----- device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_precomputed = True   # False → compute embeddings on-the-fly (slow)
    results_dir     = '../../results/transformer_lstm_v1'
    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # Load data
    # ----------------------------------------------------------------------- #
    if use_precomputed:
        print("Loading precomputed LSTM embeddings...")
        X      = torch.load('../../data/mimic_data/train_sapbert_lstm_embeddings.pt')
        masks  = torch.load('../../data/mimic_data/train_sapbert_lstm_masks.pt')
        y      = torch.load('../../data/mimic_data/train_sapbert_lstm_labels.pt')

        # train / val split — keep indices aligned across all three tensors
        indices = list(range(len(y)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        train_dataset = lstm_precomputed_dataset(
            X[train_idx], masks[train_idx], y[train_idx]
        )
        val_dataset = lstm_precomputed_dataset(
            X[val_idx], masks[val_idx], y[val_idx]
        )
    else:
        # On-the-fly tokenisation via the existing mimic_dataset
        data = pd.read_csv('../../data/mimic_data/train_data.csv')
        X_raw = data['TEXT'].tolist()
        y_raw = data['ICD9_CODE'].astype(int).values

        tokenizer = AutoTokenizer.from_pretrained(
            "../helper_code/sapBERT_local_save", local_files_only=True
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=42
        )
        train_dataset = mimic_dataset(X_train, y_train, tokenizer, precomputed=False)
        val_dataset   = mimic_dataset(X_val,   y_val,   tokenizer, precomputed=False)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True,  num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=512, shuffle=False, num_workers=0, pin_memory=False
    )

    # ----------------------------------------------------------------------- #
    # Model, loss, optimiser, scheduler
    # ----------------------------------------------------------------------- #
    model = transformer_lstm(
        lstm_hidden_size=256,
        lstm_num_layers=2,
        dropout=0.25,
        bidirectional=True,
        device=device,
        load_sapbert=(not use_precomputed),
    )
    model.to(device)

    scaler    = torch.amp.GradScaler('cuda')
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # ----------------------------------------------------------------------- #
    # Training loop
    # ----------------------------------------------------------------------- #
    history = {'train_loss': [], 'val_loss': [], 'precision': [], 'recall': [], 'f1': []}
    num_epochs = 10

    for epoch in range(num_epochs):
        # ------------------------------------------------------------------ #
        # Training
        # ------------------------------------------------------------------ #
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for x_batch, labels in train_bar:
            if use_precomputed:
                x_batch = {k: v.to(device) for k, v in x_batch.items()}
            else:
                x_batch = {k: v.to(device) for k, v in x_batch.items()}
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(x_batch)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # Gradient clipping helps stabilise LSTM training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # ------------------------------------------------------------------ #
        # Validation
        # ------------------------------------------------------------------ #
        model.eval()
        total_val_loss = 0
        tp = fp = tn = fn = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

        with torch.no_grad():
            for x_batch, labels in val_bar:
                x_batch = {k: v.to(device) for k, v in x_batch.items()}
                labels  = labels.to(device).float().view(-1, 1)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(x_batch)
                    loss    = criterion(outputs, labels)

                total_val_loss += loss.item()

                predictions = (outputs > 0).int().view(-1)
                targets     = (labels  > 0).int().view(-1)

                tp += ((predictions == 1) & (targets == 1)).sum().item()
                fp += ((predictions == 1) & (targets == 0)).sum().item()
                tn += ((predictions == 0) & (targets == 0)).sum().item()
                fn += ((predictions == 0) & (targets == 1)).sum().item()

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

        print(
            f"Epoch {epoch+1}/{num_epochs}  "
            f"LR: {current_lr:.2e}  "
            f"Train Loss: {avg_train_loss:.4f}  "
            f"Val Loss: {avg_val_loss:.4f}  "
            f"Precision: {precision:.4f}  "
            f"Recall: {recall:.4f}  "
            f"F1: {f1:.4f}"
        )

        torch.save(
            model.state_dict(),
            os.path.join(results_dir, f'model_epoch_{epoch+1}.pth')
        )

    # ----------------------------------------------------------------------- #
    # Plots
    # ----------------------------------------------------------------------- #
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'],   label='Val Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_plot.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['precision'], label='Precision')
    plt.plot(epochs_range, history['recall'],    label='Recall')
    plt.plot(epochs_range, history['f1'],        label='F1 Score')
    plt.title('Classification Metrics Over Time')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend()
    plt.savefig(os.path.join(results_dir, 'metrics_plot.png'))
    plt.close()

    print("Plots saved to", results_dir)


if __name__ == "__main__":
    main()
