import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from helper_code.mimic_dataset import mimic_dataset
from transformer_lstm import transformer_lstm


# --------------------------------------------------------------------------- #
# Dataset wrapper — mirrors the one in train.py
# --------------------------------------------------------------------------- #
class lstm_precomputed_dataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, masks, labels):
        self.embeddings = embeddings
        self.masks      = masks
        self.labels     = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            {
                'embeddings':     self.embeddings[idx],
                'attention_mask': self.masks[idx],
            },
            self.labels[idx],
        )


def main():
    # ----- device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- configuration -----
    # Update to the epoch with the best validation metrics
    model_path      = '../../results/transformer_lstm_v1/model_epoch_10.pth'
    results_dir     = '../../results/transformer_lstm_v1'
    use_precomputed = True

    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # Load test data
    # ----------------------------------------------------------------------- #
    if use_precomputed:
        print("Loading precomputed LSTM test embeddings...")
        X_test     = torch.load('../../data/mimic_data/test_sapbert_lstm_embeddings.pt')
        masks_test = torch.load('../../data/mimic_data/test_sapbert_lstm_masks.pt')
        y_test     = torch.load('../../data/mimic_data/test_sapbert_lstm_labels.pt')
        test_dataset = lstm_precomputed_dataset(X_test, masks_test, y_test)
    else:
        test_data_path = '../../data/mimic_data/test_data.csv'
        if not os.path.exists(test_data_path):
            print(f"Error: Could not find test data at {test_data_path}")
            return
        data = pd.read_csv(test_data_path)
        X_raw  = data['TEXT'].tolist()
        y_test = data['ICD9_CODE'].astype(int).values
        tokenizer    = AutoTokenizer.from_pretrained(
            "../helper_code/sapBERT_local_save", local_files_only=True
        )
        test_dataset = mimic_dataset(X_raw, y_test, tokenizer, precomputed=False)

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )

    # ----------------------------------------------------------------------- #
    # Load model
    # ----------------------------------------------------------------------- #
    model = transformer_lstm(
        lstm_hidden_size=256,
        lstm_num_layers=2,
        dropout=0.25,
        bidirectional=True,
        device=device,
        load_sapbert=(not use_precomputed),
    )

    if not os.path.exists(model_path):
        print(f"Error: Could not find model weights at {model_path}")
        return
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    # ----------------------------------------------------------------------- #
    # Evaluation loop
    # ----------------------------------------------------------------------- #
    total_test_loss = 0
    all_targets     = []
    all_predictions = []
    all_outputs     = []   # raw logits for AUC

    test_bar = tqdm(test_loader, desc="[Testing]")

    with torch.no_grad():
        for x_batch, labels in test_bar:
            x_batch = {k: v.to(device) for k, v in x_batch.items()}
            labels  = labels.to(device).float().view(-1, 1)

            outputs = model(x_batch)
            loss    = criterion(outputs, labels)
            total_test_loss += loss.item()

            predictions = (outputs > 0).int().view(-1)
            targets     = labels.int().view(-1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_outputs.extend(outputs.view(-1).cpu().numpy())

            test_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_test_loss = total_test_loss / len(test_loader)

    # ----------------------------------------------------------------------- #
    # Print metrics
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 50)
    print("TESTING RESULTS")
    print("=" * 50)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print("-" * 50)

    print("Classification Report:")
    print(classification_report(all_targets, all_predictions, zero_division=0))
    print("-" * 50)

    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions))

    try:
        roc_auc = roc_auc_score(all_targets, all_outputs)
        print("-" * 50)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        fpr, tpr, _ = roc_curve(all_targets, all_outputs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve — Transformer + LSTM')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        plot_path = os.path.join(results_dir, 'roc_curve.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"\nROC curve saved to {plot_path}")

    except ValueError:
        print("-" * 50)
        print("ROC-AUC Score: N/A (only one class present in test data)")


if __name__ == "__main__":
    main()
