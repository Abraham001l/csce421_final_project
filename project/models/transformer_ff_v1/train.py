from transformers import AutoTokenizer
from transformer_ff import transformer_ff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import os
import sys
import matplotlib.pyplot as plt
import tqdm

# prevent the tokenizer from deadlocking the Dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from helper_code.mimic_dataset import mimic_dataset

def main():
    # ----- setting device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- load the training and validation data -----
    data = pd.read_csv('../../data/mimic_data/train_data.csv')
    X = data['TEXT'].tolist()
    y = data['ICD9_CODE'].astype(int).values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----- create datasets and dataloaders -----
    tokenizer = AutoTokenizer.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
    train_dataset = mimic_dataset(X_train, y_train, tokenizer)
    val_dataset = mimic_dataset(X_val, y_val, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # ----- initialize the model, loss function, and optimizer -----
    model = transformer_ff(device=device)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # ----- training and validation loop -----
    # lists to store metrics for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'f1': [] 
    }
    num_epochs = 10
    for epoch in range(num_epochs):
        # ---- training loop -----
        model.train()
        total_train_loss = 0

        # tqdm for progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for texts_embeddings, labels in train_loader:
            # move data to device
            texts_embeddings = {key: val.to(device) for key, val in texts_embeddings.items()}
            labels = labels.to(device).view(-1, 1)  # reshape labels to (batch_size, 1)

            # forward pass
            optimizer.zero_grad()
            outputs = model(texts_embeddings)
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # ----- validation loop -----
        model.eval()
        total_val_loss = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        # tqdm for progress bar
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

        with torch.no_grad():
            for texts_embeddings, labels in val_loader:
                texts_embeddings = {key: val.to(device) for key, val in texts_embeddings.items()}
                labels = labels.to(device).view(-1, 1)
                outputs = model(texts_embeddings)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                # getting prediction and target (flattening with .view(-1))
                predictions = (outputs > 0).int().view(-1)
                targets = (labels > 0).int().view(-1)

                # updating confusion matrix counts
                tp += ((predictions == 1) & (targets == 1)).sum().item()
                fp += ((predictions == 1) & (targets == 0)).sum().item()
                tn += ((predictions == 0) & (targets == 0)).sum().item()
                fn += ((predictions == 0) & (targets == 1)).sum().item()

                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)

        # calculating precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # storing metrics for plotting
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}")

        model_path = f'../../results/transformer_ff_v1/model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_path)

    # ----- plotting the metrics -----
    epochs_range = range(1, num_epochs + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../../results/transformer_ff_v1/loss_plot.png')

    # Plot Metrics
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['precision'], label='Precision')
    plt.plot(epochs_range, history['recall'], label='Recall')
    plt.plot(epochs_range, history['f1'], label='F1 Score')
    plt.title('Classification Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('../../results/transformer_ff_v1/metrics_plot.png')

    print("Plots saved as loss_plot.png and metrics_plot.png")

if __name__ == "__main__":
    # avoiding child process spawning issues on Windows
    main()
