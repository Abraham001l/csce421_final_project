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
from tqdm import tqdm
# prevent the tokenizer from deadlocking the Dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from helper_code.mimic_dataset import mimic_dataset

def main():
    # ----- setting device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_precomputed = True  # Set to False to compute embeddings on the fly (not recommended for large datasets)  

    # ----- load the training and validation data -----
    X, y = None, None
    if use_precomputed:
        print("Loading precomputed embeddings...")
        X = torch.load('../../data/mimic_data/train_sapbert_embeddings.pt')
        y = torch.load('../../data/mimic_data/train_sapbert_labels.pt')
    else:
        data = pd.read_csv('../../data/mimic_data/train_data.csv')
        X = data['TEXT'].tolist()
        y = data['ICD9_CODE'].astype(int).values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----- create datasets and dataloaders -----
    tokenizer = AutoTokenizer.from_pretrained("../helper_code/sapBERT_local_save", local_files_only=True)
    train_dataset = mimic_dataset(X_train, y_train, tokenizer, precomputed=use_precomputed)
    val_dataset = mimic_dataset(X_val, y_val, tokenizer, precomputed=use_precomputed)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    # ----- initialize the model, loss function, and optimizer -----
    model = transformer_ff(
        dropout=0.2, 
        hidden1_size=256, 
        hidden2_size=128, 
        device=device,
        load_sapbert=(not use_precomputed)
    )
    model.to(device)
    scaler = torch.amp.GradScaler('cuda')  # for mixed precision training
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

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
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for texts_embeddings, labels in train_bar:
            # move data to device
            if (not use_precomputed):
                texts_embeddings = {key: val.to(device) for key, val in texts_embeddings.items()}
            else:
                texts_embeddings = texts_embeddings.to(device)
            labels = labels.to(device).view(-1, 1)  # reshape labels to (batch_size, 1)

            # forward pass
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(texts_embeddings)
                loss = criterion(outputs, labels)

            # backward pass
            scaler.scale(loss).backward() # scale up loss to keep gradient in range of float16
            scaler.step(optimizer) # unscale gradients to update weights properly
            scaler.update() # update the scale factor for next iteration

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # ----- validation loop -----
        model.eval()
        total_val_loss = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        # tqdm for progress bar
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

        with torch.no_grad():
            for texts_embeddings, labels in val_bar:
                if (not use_precomputed):
                    texts_embeddings = {key: val.to(device) for key, val in texts_embeddings.items()}
                else:
                    texts_embeddings = texts_embeddings.to(device)
                labels = labels.to(device).view(-1, 1)
                with torch.amp.autocast(device_type=device.type):
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
        scheduler.step(avg_val_loss)  # adjust learning rate based on validation loss
        current_lr = optimizer.param_groups[0]['lr']

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

        print(f"Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}")

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
